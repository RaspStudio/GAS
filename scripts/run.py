#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import subprocess
import shutil
from dataclasses import dataclass
from itertools import product
from datetime import datetime
from typing import Union, List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# -----------------------------------------------------------------------------
# Data definitions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    name: str
    dim: int
    data_path: str
    query_path: str
    max_elements: Union[int, List[int]]
    max_queries: Union[int, List[int]]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]

def _short_path(p: str) -> str:
    base = os.path.basename(os.path.expanduser(p))
    name, _ = os.path.splitext(base)
    return name or base

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _normalize_methods(
    methods: Optional[Union[
        int,
        List[int],
        Dict[str, Union[Optional[int], Tuple[Optional[int], List[int]], Dict[str, Any]]]
    ]]
) -> List[Tuple[str, Optional[int], List[int]]]:
    if methods is None:
        return [("all", None, [])]
    if isinstance(methods, int):
        return [(f"idx{methods}", methods, [])]
    if isinstance(methods, list):
        return [(f"idx{int(v)}", int(v), []) for v in methods]
    if isinstance(methods, dict):
        def _extract(val) -> Tuple[Optional[int], List[int]]:
            if isinstance(val, tuple) and len(val) == 2:
                idx, efs = val
                return (None if idx is None else int(idx), [int(x) for x in _as_list(efs)])
            if isinstance(val, dict):
                idx = val.get("idx", None)
                efs = val.get("efs", [])
                return (None if idx is None else int(idx), [int(x) for x in _as_list(efs)])
            return (None if val is None else int(val), [])
        ids = []
        for v in methods.values():
            idx, _efs = _extract(v)
            if idx is not None:
                ids.append(idx)
        if len(ids) != len(set(ids)):
            raise ValueError("methods dict has duplicate only_run_idx values.")
        out: List[Tuple[str, Optional[int], List[int]]] = []
        for name, val in sorted(methods.items(), key=lambda x: x[0]):
            idx, efs = _extract(val)
            out.append((str(name), idx, efs))
        return out
    raise TypeError("methods must be None/int/list[int]/dict[str, Optional[int]]")

def _append_positional_tail(cmd: List[str],
                            only_run_idx: Optional[int],
                            repeat: Optional[int],
                            n_seg: Optional[int],
                            ef_list: List[int]) -> None:
    have_ef = bool(ef_list)

    # 1) only_run_idx（argv[9]）
    if only_run_idx is not None:
        cmd.append(str(only_run_idx))
        # 2) repeat（argv[10]）
        if repeat is not None:
            cmd.append(str(repeat))
        elif (n_seg is not None) or have_ef:
            cmd.append("1") 
        # 3) n_seg（argv[11]）
        if n_seg is not None:
            cmd.append(str(n_seg))
        elif have_ef:
            cmd.append("1")  
    else:
        if (repeat is not None) or (n_seg is not None) or have_ef:
            cmd.append(str(0xFFFFFFFF))  # only_run_idx 
            # 2) repeat（argv[10]）
            if repeat is not None:
                cmd.append(str(repeat))
            else:
                cmd.append("1")  # 
            # 3) n_seg（argv[11]）
            if n_seg is not None:
                cmd.append(str(n_seg))
            elif have_ef:
                cmd.append("1")  # 

    # 4) efs...
    if have_ef:
        cmd.extend(str(v) for v in ef_list)

# -----------------------------------------------------------------------------
# NUMA helpers
# -----------------------------------------------------------------------------

def _expand_cpulist(spec: str) -> List[int]:
    out: List[int] = []
    spec = spec.strip()
    if not spec:
        return out
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

def _compress_cpulist(cpus: List[int]) -> str:
    if not cpus:
        return ""
    cpus = sorted(set(cpus))
    runs = []
    s = e = cpus[0]
    for x in cpus[1:]:
        if x == e + 1:
            e = x
        else:
            runs.append((s, e))
            s = e = x
    runs.append((s, e))
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)

def detect_numa_nodes() -> List[Tuple[int, str, List[int]]]:
    """Return list of (node_id, cpulist_str, cpus_array)."""
    base = Path("/sys/devices/system/node")
    nodes: List[Tuple[int, str, List[int]]] = []
    if not base.is_dir():
        all_cpus = list(range(os.cpu_count() or 1))
        return [(0, _compress_cpulist(all_cpus), all_cpus)]
    for p in sorted(base.iterdir()):
        m = re.match(r"node(\d+)$", p.name)
        if not m:
            continue
        nid = int(m.group(1))
        cpulist_path = p / "cpulist"
        if not cpulist_path.exists():
            continue
        s = cpulist_path.read_text().strip()
        cpus = _expand_cpulist(s)
        if cpus:
            nodes.append((nid, s, cpus))
    if not nodes:
        all_cpus = list(range(os.cpu_count() or 1))
        return [(0, _compress_cpulist(all_cpus), all_cpus)]
    return nodes

# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def run_experiment_batch(
    bench_exec_path: str,
    datasets: Union[Dataset, List[Dataset]],
    k: Union[int, List[int]],
    bmeta_path: Union[str, List[str], Dict[str, Union[str, List[str]]]],
    qmeta_path: Union[str, List[str], Dict[str, Union[str, List[str]]]],
    methods: Optional[Union[
        int,
        List[int],
        Dict[str, Union[Optional[int], Tuple[Optional[int], List[int]], Dict[str, Any]]]
    ]] = None,
    repeat: Optional[int] = None,
    n_seg: Optional[int] = None,          
    log_root_dir: str = "logs",
    group_name: Optional[str] = None,
    max_threads: int = 1,
    # NUMA binding options
    bind_numa: bool = True,
    allowed_numa_nodes: Optional[List[int]] = None,  
    cpus_per_task: Optional[int] = None,             
    set_omp_threads: bool = True,                
) -> None:
    """
    CLI:
      ./bench dim max_elements max_queries k data_path bmeta_path query_path qmeta_path [only_run_idx] [repeat] [n_seg] [efs...]

    Structure:
      <log_root_dir>/<YYYYMMDD_HHMMSS>[_<group_name>]/
        <dataset_name>/
          k<k>-bmeta_<b>-qmeta_<q>[-seg<n_seg>]/
            m_<method_name>/
              n_<max_elements>_q_<max_queries>_<timestamp>.log
    """
    ds_list: List[Dataset] = _as_list(datasets)
    k_list = _as_list(k)
    method_triples = _normalize_methods(methods)  # [(method_name, only_run_idx_or_None, ef_list), ...]

    os.makedirs(log_root_dir, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{date_tag}" if not group_name else f"{date_tag}_{_slug(group_name)}"
    run_dir = os.path.join(log_root_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    bench_exec_path_new = os.path.join(run_dir, "bench")
    if not os.path.exists(bench_exec_path_new):
        try:
            shutil.copy(bench_exec_path, bench_exec_path_new)
        except Exception as e:
            print(f"❌ Failed to copy executable: {e}")
            raise
    bench_exec_path = bench_exec_path_new

    numa_nodes = detect_numa_nodes()  # [(nid, cpulist_str, cpus_array), ...]
    if allowed_numa_nodes is not None:
        numa_nodes = [t for t in numa_nodes if t[0] in set(allowed_numa_nodes)]
        if not numa_nodes:
            raise RuntimeError("allowed_numa_nodes filtered out all available NUMA nodes.")

    def _slice_cpulist_for_task(cpus_array: List[int]) -> str:
        if cpus_per_task is None or cpus_per_task >= len(cpus_array):
            return _compress_cpulist(cpus_array)
        return _compress_cpulist(cpus_array[:cpus_per_task])

    jobs: List[dict] = []
    for ds in ds_list:
        me_list = _as_list(ds.max_elements)
        mq_list = _as_list(ds.max_queries)

        if isinstance(bmeta_path, dict):
            if ds.name not in bmeta_path:
                raise ValueError(f"bmeta_path dict missing key for dataset '{ds.name}'")
            b_list = _as_list(bmeta_path[ds.name])
        else:
            b_list = _as_list(bmeta_path)

        if isinstance(qmeta_path, dict):
            if ds.name not in qmeta_path:
                raise ValueError(f"qmeta_path dict missing key for dataset '{ds.name}'")
            q_list = _as_list(qmeta_path[ds.name])
        else:
            q_list = _as_list(qmeta_path)

        for me, mq, kk, bm, qm, (mname, midx, mefs) in product(me_list, mq_list, k_list, b_list, q_list, method_triples):
    
            if bind_numa and numa_nodes:
                idx = len(jobs)
                nid, _cpulist_full, cpus_arr = numa_nodes[idx % len(numa_nodes)]
                cpulist_str = _slice_cpulist_for_task(cpus_arr)
            else:
                nid, cpulist_str = None, ""

            jobs.append({
                "dataset": ds,
                "max_elements": me,
                "max_queries": mq,
                "k": kk,
                "bmeta_path": bm,
                "qmeta_path": qm,
                "method_name": mname,   
                "only_run_idx": midx,    
                "efs": mefs,             
                "numa_node": nid,         
                "cpu_list_str": cpulist_str, 
            })

    def run_single(job: dict) -> str:
        ds: Dataset = job["dataset"]
        only_run_idx = job["only_run_idx"]
        ef_list: List[int] = job.get("efs", [])

        cmd = [
            bench_exec_path,
            str(ds.dim),
            str(job["max_elements"]),
            str(job["max_queries"]),
            str(job["k"]),
            os.path.expanduser(ds.data_path),
            os.path.expanduser(job["bmeta_path"]),
            os.path.expanduser(ds.query_path),
            os.path.expanduser(job["qmeta_path"]),
        ]

        _append_positional_tail(
            cmd,
            only_run_idx=only_run_idx,
            repeat=repeat,
            n_seg=n_seg,
            ef_list=ef_list
        )

        ds_dir = os.path.join(run_dir, _slug(ds.name))
        os.makedirs(ds_dir, exist_ok=True)

        b_tag = _slug(_short_path(job["bmeta_path"]))
        q_tag = _slug(_short_path(job["qmeta_path"]))
        seg_tag = f"-seg{n_seg}" if n_seg is not None else ""
        setting_dir = os.path.join(ds_dir, f"k{job['k']}-bmeta_{b_tag}-qmeta_{q_tag}{seg_tag}")
        os.makedirs(setting_dir, exist_ok=True)

        method_dir = os.path.join(setting_dir, f"m_{_slug(job['method_name'])}")
        os.makedirs(method_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"n_{job['max_elements']}_q_{job['max_queries']}_{ts}.log"
        log_path = os.path.join(method_dir, log_filename)

        prefix: List[str] = []
        if bind_numa and job["numa_node"] is not None:
            nid = job["numa_node"]
            cpu_list_str = job["cpu_list_str"]
            numactl_path = shutil.which("numactl")
            if numactl_path:
                prefix = [numactl_path, f"--physcpubind={cpu_list_str}", f"--membind={nid}"]
            else:
                tpath = shutil.which("taskset")
                if tpath:
                    print("⚠️  numactl not found, using taskset instead for CPU binding only")
                    prefix = [tpath, "-c", cpu_list_str]
                else:
                    print("⚠️  numactl/taskset are not found, running without CPU/NUMA binding")

        full_cmd = prefix + cmd
        env = os.environ.copy()
        if bind_numa and set_omp_threads and job["numa_node"] is not None:
            if "OMP_NUM_THREADS" not in env and job["cpu_list_str"]:
                omp_threads = 1 
                if omp_threads > 0:
                    env["OMP_NUM_THREADS"] = str(omp_threads)
                    env.setdefault("GOMP_CPU_AFFINITY", job["cpu_list_str"])
                    env.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

        print(f"▶️ Running: {' '.join(full_cmd)}")
        try:
            with open(log_path, "w") as logf:
                completed = subprocess.run(full_cmd, stdout=logf, stderr=subprocess.STDOUT,
                                           check=False, env=env)
            if completed.returncode != 0:
                print(f"⚠️ Return code {completed.returncode}: {log_path}")
        except Exception as e:
            print(f"❌ Failed: {log_path} ({e})")
            raise
        return log_path

    with ThreadPoolExecutor(max_workers=max_threads) as ex:
        futures = [ex.submit(run_single, job) for job in jobs]
        for fu in as_completed(futures):
            log_path = fu.result()
            print(f"✅ Finished: {log_path}")

    print(f"🎉 All experiments completed. Logs saved under: {run_dir}")

# -----------------------------------------------------------------------------
# Build helper
# -----------------------------------------------------------------------------

def build_project(
    build_dir: str,
    *,
    use_parent_as_source: bool = True,         
    cmake_build_type: str = "Release",
    extra_cmake_args: Optional[List[str]] = None,
    clean: bool = False,                       
    jobs: Optional[int] = None,               
    build_tool: str = "make",                   
    build_targets: Optional[List[str]] = None, 
    build_log_dir: Optional[str] = None,        
) -> str:

    os.makedirs(build_dir, exist_ok=True)

    if clean:
        cache = os.path.join(build_dir, "CMakeCache.txt")
        cm_files = os.path.join(build_dir, "CMakeFiles")
        for p in (cache,):
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
        if os.path.isdir(cm_files):
            shutil.rmtree(cm_files, ignore_errors=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (os.path.join(build_log_dir, f"build_{ts}.log")
                if build_log_dir else os.path.join(build_dir, f"build_{ts}.log"))
    if build_log_dir:
        os.makedirs(build_log_dir, exist_ok=True)

    cmake_cmd = ["cmake", f"-DCMAKE_BUILD_TYPE={cmake_build_type}"]
    if extra_cmake_args:
        cmake_cmd.extend(extra_cmake_args)
    cmake_cmd.append(".." if use_parent_as_source else ".")

    if jobs is None:
        jobs = max(1, (os.cpu_count() or 1))
    build_cmd = [build_tool]
    if build_tool in ("make", "ninja"):
        build_cmd += ["-j", str(jobs)]
    if build_targets:
        build_cmd += build_targets

    with open(log_path, "w") as logf:
        logf.write(f"$ {' '.join(cmake_cmd)}")
        logf.flush()
        subprocess.run(cmake_cmd, cwd=build_dir, stdout=logf, stderr=subprocess.STDOUT, check=True)

        logf.write(f"$ {' '.join(build_cmd)}")
        logf.flush()
        subprocess.run(build_cmd, cwd=build_dir, stdout=logf, stderr=subprocess.STDOUT, check=True)

    print(f"🏗️ Build completed. Log: {log_path}")
    return log_path

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Datasets Example
    SIFT_1M_EXAMPLE = "Sift1M"
    SIFT_1M_EXAMPLE_DATASET = Dataset(
        name=SIFT_1M_EXAMPLE,
        dim=128,
        data_path="path.fvecs",
        query_path="path.fvecs",
        max_elements=[1_000_000],
        max_queries=[1],
    )


    build_project(
        build_dir="./build",
        use_parent_as_source=True,
        cmake_build_type="Release",
        extra_cmake_args=["-DDEFINE_RUN_SCRIPT_BUILD=ON"],
        clean=False,
        jobs=16,
        build_tool="make",
        build_targets=None,
        build_log_dir="./log",
    )


    run_experiment_batch(
        bench_exec_path="./build/bench",
        datasets=[
            SIFT_1M_EXAMPLE_DATASET,
        ],
        k=[10],
        bmeta_path={
            SIFT_1M_EXAMPLE: "path.bmeta",
        },
        qmeta_path={
            SIFT_1M_EXAMPLE: ["path.qmeta"],
        },
        methods={
            "std-gas": (1, [10]),
        },
        repeat=1,
        log_root_dir="./log",
        group_name="example",
        n_seg=1,
        max_threads=1,
        bind_numa=True,
        allowed_numa_nodes=None, 
        cpus_per_task=None,   
        set_omp_threads=True,
    )
