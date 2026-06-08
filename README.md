# GAS Library

A lightweight framework for filtered search built on [hnswlib](https://github.com/nmslib/hnswlib).

## Quick Start (WIT Example)

```bash
cd example
./run_wit.sh                           # auto-download dataset + run benchmark
./run_wit.sh --wit-dir /path/to/parquets  # use pre-downloaded parquet files
```

This downloads 6 parquet parts from `wikimedia/wit_base` (~117k rows), extracts 100k base + 1k query vectors (2048d), and runs the benchmark.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Usage

```cpp
#include "gaslib.h"

gaslib::FvecsDatasetWithMeta dataset(dim, max_elements, data_path, bmeta_path);
auto idx = gaslib::GasIndex<decltype(dataset), gaslib::RangeGasFilterFunctor, 2, true>(dataset, 4, 12);
```

## Run Benchmarks

```bash
./bench dim max_elements max_queries k cache_dir data_path bmeta_path query_path qmeta_path \
        [only_run_idx] [repeat] [n_seg] [batch_size] [query_seq_mode] [ef ...]
```