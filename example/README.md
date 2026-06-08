# WIT Example

## Prerequisites

- **System**: cmake, make, g++, wget
- **Python**: pyarrow, numpy (`pip install pyarrow numpy`)
- **Disk**: ~1.2 GB (6 parquet files, ~180 MB each)

## Run

```bash
./run_wit.sh                          # download + extract + build + benchmark
./run_wit.sh --wit-dir /path/to/dir   # skip download, use existing parquets
```

## What It Does

| Step | Description |
|------|-------------|
| Download | 6 parquet parts from `wikimedia/wit_base` (~180 MB each) |
| Extract | 100k base + 1k query vectors (2048-dim), auto-select ~4% selectivity filter |
| Build | Compile `bench` and `gengt` targets |
| Ground truth | Generate exact k-NN results with brute-force index |
| Benchmark | Run GAS indexes and print recall/latency tables |

## Expected Output

Latency may vary based on hardware, but hops, distance computations, and recall should be similar to below:

| Method | ef | Hops | Distance Computations | Recall | Latency (us) |
|--------|----|------|----------------------|--------|-------------|
| HNSWlib-Baseline | 10 | 6,960,121 | 21,807,393 | 0.7976 | 20,754 |
| HNSWlib-Baseline | 15 | 7,579,985 | 23,292,969 | 0.8508 | 22,789 |
| HNSWlib-Baseline | 20 | 8,065,205 | 24,431,586 | 0.8810 | 23,702 |
| HNSWlib-Baseline | 30 | 8,806,094 | 26,160,323 | 0.9138 | 25,528 |
| HNSW-Baseline | 10 | 6,961,115 | 21,807,393 | 0.7976 | 21,673 |
| HNSW-Baseline | 15 | 7,580,979 | 23,292,969 | 0.8508 | 22,914 |
| HNSW-Baseline | 20 | 8,066,199 | 24,431,586 | 0.8810 | 23,967 |
| HNSW-Baseline | 30 | 8,807,088 | 26,160,323 | 0.9138 | 26,091 |
| GAS-Opt (AE2/4, SC4/12) | 10 | 6,618,073 | 21,158,761 | 0.8886 | 22,089 |
| GAS-Opt (AE2/4, SC4/12) | 15 | 7,236,668 | 22,180,622 | 0.9273 | 23,192 |
| GAS-Opt (AE2/4, SC4/12) | 20 | 7,706,600 | 22,466,424 | 0.9457 | 23,923 |
| GAS-Opt (AE2/4, SC4/12) | 30 | 8,391,960 | 22,735,051 | 0.9646 | 24,399 |
