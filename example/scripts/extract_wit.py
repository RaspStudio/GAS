#!/usr/bin/env python3
"""
Extract embedding vectors + mime_type metadata from WIT parquet files,
output as fvecs + bmeta + qmeta in one pass.

Usage:
    python3 extract_wit.py --parquet-dir /path/to/parquets

Input:
    train-00000~00005-of-00330.parquet (first 6 parts by default)

Output:
    temp/wit_base.fvecs              (base vectors, fvecs format)
    temp/wit_base.bmeta              (base metadata, attribute IDs as int32)
    temp/wit_query.fvecs             (query vectors, fvecs format)
    temp/wit_query_range.qmeta       (query filter, ~4% selectivity range filter)
"""

import struct
import os
import sys
import argparse
from collections import Counter
import numpy as np
import pyarrow.parquet as pq

# ============================================================
# Defaults (overridable via CLI)
# ============================================================
NUM_PARTS = 6                      # first 6 parts (~117k rows, enough for 101k)
BASE_COUNT = 100000                # base vectors
QUERY_COUNT = 1000                 # query vectors
TOTAL_NEED = BASE_COUNT + QUERY_COUNT

# mime_type -> int32 ID mapping
MIME_TO_ID = {
    "image/jpeg":     0,
    "image/png":      1,
    "image/svg+xml":  2,
    "image/gif":      3,
}
DEFAULT_MIME_ID = 0    # None values default to jpeg

# target selectivity for the range filter
TARGET_SELECTIVITY = 0.04  # 4%

# —— Resolve paths ——
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "temp")

OUT_BASE_FVECS  = os.path.join(OUT_DIR, "wit_base.fvecs")
OUT_BASE_BMETA  = os.path.join(OUT_DIR, "wit_base.bmeta")
OUT_QUERY_FVECS = os.path.join(OUT_DIR, "wit_query.fvecs")
OUT_QUERY_QMETA = os.path.join(OUT_DIR, "wit_query_range.qmeta")


def write_fvecs(path: str, vectors: np.ndarray) -> None:
    """Write numpy array (n, dim) as fvecs file."""
    n, dim = vectors.shape
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", dim))
            f.write(vectors[i].astype(np.float32).tobytes())
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[fvecs] wrote {path}  ({n} x {dim}, {size_mb:.1f} MB)")


def write_bmeta(path: str, meta_ids: np.ndarray) -> None:
    """Write int32 metadata array as bmeta file."""
    with open(path, "wb") as f:
        for v in meta_ids:
            f.write(struct.pack("<i", int(v)))
    size_kb = os.path.getsize(path) / 1024
    print(f"[bmeta] wrote {path}  ({len(meta_ids)} entries, {size_kb:.1f} KB)")


def write_qmeta_range(path: str, n_queries: int, start: int, end: int) -> None:
    """Write range-filter qmeta file (all queries share the same range)."""
    with open(path, "wb") as f:
        for _ in range(n_queries):
            f.write(struct.pack("<i", -1))   # range marker
            f.write(struct.pack("<i", start))
            f.write(struct.pack("<i", end))
    size_kb = os.path.getsize(path) / 1024
    print(f"[qmeta] wrote {path}  ({n_queries} entries, range=[{start},{end}], {size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embedding vectors + metadata from WIT parquet -> fvecs + bmeta + qmeta")
    parser.add_argument("parquet_dir", help="directory containing downloaded parquet files")
    parser.add_argument("out_dir", nargs="?", default=OUT_DIR,
                        help=f"output directory (default: {OUT_DIR})")
    args = parser.parse_args()

    parquet_dir = args.parquet_dir
    num_parts   = NUM_PARTS
    base_count  = BASE_COUNT
    query_count = QUERY_COUNT
    total_need  = base_count + query_count
    out_dir     = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    out_base_fvecs  = os.path.join(out_dir, "wit_base.fvecs")
    out_base_bmeta  = os.path.join(out_dir, "wit_base.bmeta")
    out_query_fvecs = os.path.join(out_dir, "wit_query.fvecs")
    out_query_qmeta = os.path.join(out_dir, "wit_query_range.qmeta")

    dim = None
    all_vectors = []   # list of numpy arrays (float32)
    all_mimes   = []   # list of str or None
    collected = 0
    part_index = 0

    print(f"Extracting {num_parts} parts: {base_count} base + {query_count} query, "
          f"filter target ~{TARGET_SELECTIVITY*100:.0f}%")

    while collected < total_need and part_index < num_parts:
        part_name = f"train-{part_index:05d}-of-00330.parquet"
        part_path = os.path.join(parquet_dir, part_name)

        if not os.path.exists(part_path):
            print(f"[skip] file not found: {part_path}")
            part_index += 1
            continue

        pf = pq.ParquetFile(part_path)

        if dim is None:
            batch = pf.read_row_group(0, columns=["embedding"])
            dim = len(batch.column("embedding")[0].as_py())

        for rg_idx in range(pf.metadata.num_row_groups):
            if collected >= total_need:
                break

            need = total_need - collected
            batch = pf.read_row_group(rg_idx, columns=["embedding", "mime_type"])
            n_in_rg = batch.num_rows
            take = min(n_in_rg, need)

            emb_list = batch.column("embedding").to_pylist()
            chunk_emb = emb_list[:take]
            mime_list = batch.column("mime_type").to_pylist()
            chunk_mime = mime_list[:take]

            arr = np.array(chunk_emb, dtype=np.float32)
            all_vectors.append(arr)
            all_mimes.extend(chunk_mime)
            collected += take

            if collected % 50000 < take:
                print(f"  {collected:,} / {total_need:,} rows")

        part_index += 1

    data = np.concatenate(all_vectors, axis=0).astype(np.float32)

    # ---- mime_type -> int32 ID ----
    mime_ids = []
    for m in all_mimes:
        mime_ids.append(MIME_TO_ID.get(m, DEFAULT_MIME_ID) if m else DEFAULT_MIME_ID)
    mime_ids = np.array(mime_ids, dtype=np.int32)

    # ---- Split base / query ----
    base_vecs  = data[:base_count]
    base_mimes = mime_ids[:base_count]
    query_vecs  = data[base_count:base_count + query_count]
    query_mimes = mime_ids[base_count:base_count + query_count]

    # ---- Pick filter_id closest to target selectivity in base set ----
    counter = Counter(base_mimes)
    total_base = len(base_mimes)
    best_id, best_diff = None, float("inf")
    for mid in sorted(counter.keys()):
        diff = abs(counter[mid] / total_base - TARGET_SELECTIVITY)
        if diff < best_diff:
            best_diff, best_id = diff, mid
    filter_id = best_id
    filter_pct = counter[filter_id] / total_base * 100

    write_fvecs(out_base_fvecs, base_vecs)
    write_bmeta(out_base_bmeta, base_mimes)
    write_fvecs(out_query_fvecs, query_vecs)
    write_qmeta_range(out_query_qmeta, query_count, filter_id, filter_id)

    matched = (base_mimes == filter_id).sum()
    print(f"Done: base={base_count} query={query_count} dim={dim} | "
          f"filter_id={filter_id} matched {matched}/{base_count} ({filter_pct:.1f}%)")


if __name__ == "__main__":
    main()
