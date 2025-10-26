# GAS Library

A lightweight and adaptive framework for Filtered Approximate Nearest Neighbor Search.

The implementation is build upon [hnswlib](https://github.com/nmslib/hnswlib).

## Usage Example 

See `test/benchmark.cpp` for a complete example.

```cpp
// 1. Include the header
#include "gaslib.h"

// 2. Instantiate the dataset
gaslib::FvecsDatasetWithMeta dataset(
    dim, max_elements, data_path, bmeta_path
);

// 3. Instantiate the index
auto idx = gaslib::GASHNSW<DatasetT, FilterT>>(dataset);
```

## Run Tests
To run the tests, navigate to the root directory and execute:

```bash
mkdir build
cd build
cmake ..
make
```

Then, run the tests with:

```bash
./bench dim max_elements max_queries k data_path bmeta_path query_path qmeta_path [only_run_idx] [repeat] [n_seg] [efs...]
```