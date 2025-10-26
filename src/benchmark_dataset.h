#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <set>
#include <cstdint>
#include <cstdlib>

#include "benchmark.h"   // for meta_t, label_t
#include "filter.h"    

namespace gaslib {

// Load fvecs file (binary: int32 dim, then dim floats per vector)
inline auto load_fvecs(const std::string& path, size_t dim, size_t max_elements, size_t& n_elements) -> std::vector<float> {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int dim_from_file = 0;
    in.read(reinterpret_cast<char*>(&dim_from_file), sizeof(int));
    if (dim_from_file != static_cast<int>(dim)) {
        std::cerr << "Error: Dimension mismatch. Expected " << dim << ", got " << dim_from_file << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Determine number of vectors
    in.seekg(0, std::ios::end);
    const size_t fsize = static_cast<size_t>(in.tellg());
    const int total_vecs = static_cast<int>(fsize / (sizeof(float) * (dim + 1)));

    n_elements = std::min(static_cast<size_t>(total_vecs), max_elements);
    std::vector<float> data(n_elements * dim);
    in.seekg(0, std::ios::beg);

    for (size_t i = 0; i < n_elements; ++i) {
        in.seekg(sizeof(int), std::ios::cur);  // skip dimension
        in.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
        if (!in) {
            std::cerr << "Error: Failed to read vector data at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    in.close();
    return data;
}

// Load fvecs file, each vector is 64KB aligned (binary: int32 dim, then dim floats per vector)
inline void load_fvecs_aligned(const std::string& path,
                               size_t dim, size_t pitch,
                               size_t max_elements,
                               size_t& n_elements_out,
                               float** data_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    
    in.seekg(0, std::ios::end);
    const size_t fsize = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    
    
    size_t capacity = (fsize / (sizeof(int) + dim * sizeof(float)));
    if (capacity == 0) {
        std::cerr << "Error: Empty or invalid fvecs: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    capacity = std::min(capacity, max_elements);

    
    float* buf = nullptr;
#if defined(_ISOC11_SOURCE)
    
    buf = (float*)aligned_alloc(64, pitch * capacity * sizeof(float));
    if (!buf) { std::cerr << "aligned_alloc failed\n"; std::exit(EXIT_FAILURE); }
#else
    if (posix_memalign((void**)&buf, 64, pitch * capacity * sizeof(float)) != 0 || !buf) {
        std::cerr << "posix_memalign failed\n"; std::exit(EXIT_FAILURE);
    }
#endif

    
    size_t n = 0;
    while (n < capacity && in.peek() != EOF) {
        int dim_i = 0;
        in.read(reinterpret_cast<char*>(&dim_i), sizeof(int));
        if (!in) break;
        if (dim_i != static_cast<int>(dim)) {
            std::cerr << "Error: Dimension mismatch at vec " << n
                      << ". Expected " << dim << ", got " << dim_i << std::endl;
            std::exit(EXIT_FAILURE);
        }
        float* dst = buf + n * pitch; 
        in.read(reinterpret_cast<char*>(dst), dim * sizeof(float));
        if (!in) {
            std::cerr << "Error: Failed to read vector data at index " << n << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        if (pitch > dim) {
            memset(dst + dim, 0, (pitch - dim) * sizeof(float));
        }
        ++n;
    }

    n_elements_out = n;
    *data_out = buf;
}

// Load binary metadata for dataset
inline auto load_bmeta(const std::string& meta_path, size_t n_elements) -> std::vector<meta_t> {
    std::ifstream in(meta_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open metadata file " << meta_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<meta_t> meta_data(n_elements);
    for (size_t i = 0; i < n_elements; ++i) {
        meta_t m{};
        in.read(reinterpret_cast<char*>(&m), sizeof(meta_t));
        if (!in) {
            std::cerr << "Error: Metadata file ended unexpectedly at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
        meta_data[i] = m;
    }
    in.close();
    return meta_data;
}

// Detect query metadata format by reading first int32 header
inline bool qmeta_is_range(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    int32_t header = 0;
    in.read(reinterpret_cast<char*>(&header), sizeof(int32_t));
    in.close();
    return header == -1;
}

inline bool qmeta_is_tag(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    int32_t header = 0;
    in.read(reinterpret_cast<char*>(&header), sizeof(int32_t));
    in.close();
    return header != -1;
}

// Default load_qmeta: not implemented for other filter types
template <typename filter_t>
inline auto load_qmeta(const std::string& path, size_t n_elements) -> std::vector<filter_t> {
    static_assert(sizeof(filter_t) == 0, "load_qmeta not implemented for this filter type");
    return {};
}

// Range filter specialization: file format [-1, start, end] repeated
template <>
inline auto load_qmeta<RangeGASFilterFunctor>(const std::string& path, size_t n_elements)
    -> std::vector<RangeGASFilterFunctor> {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open range metadata file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::vector<RangeGASFilterFunctor> filters;
    filters.reserve(n_elements);
    for (size_t i = 0; i < n_elements; ++i) {
        int32_t marker = 0;
        in.read(reinterpret_cast<char*>(&marker), sizeof(int32_t));
        if (!in || marker != -1) {
            std::cerr << "Error: Invalid range marker at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
        meta_t start = 0, end = 0;
        in.read(reinterpret_cast<char*>(&start), sizeof(meta_t));
        in.read(reinterpret_cast<char*>(&end), sizeof(meta_t));
        if (!in) {
            std::cerr << "Error: Range metadata file ended unexpectedly at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
        filters.emplace_back(start, end);
    }
    in.close();
    return filters;
}

// Tag filter specialization: ivecs format [len, s1, s2, ...]
template <>
inline auto load_qmeta<TagGASFilterFunctor>(const std::string& path, size_t n_elements)
    -> std::vector<TagGASFilterFunctor> {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open tag metadata file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::vector<TagGASFilterFunctor> filters;
    filters.reserve(n_elements);
    for (size_t i = 0; i < n_elements; ++i) {
        int32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
        if (!in || len < 0) {
            std::cerr << "Error: Invalid tag length at index " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
        std::set<meta_t> s;
        for (int32_t j = 0; j < len; ++j) {
            meta_t v = 0;
            in.read(reinterpret_cast<char*>(&v), sizeof(meta_t));
            if (!in) {
                std::cerr << "Error: Tag metadata ended unexpectedly at element " << j
                          << " of index " << i << std::endl;
                std::exit(EXIT_FAILURE);
            }
            s.insert(v);
        }
        filters.emplace_back(s);
    }
    in.close();
    return filters;
}

// Dataset without labels
class FvecsDataset {
public:
    FvecsDataset(size_t dim, size_t max_elements, const std::string& path)
        : dim_(dim), pitch_(((dim + 15) / 16) * 16), 
          n_elements_(0), data_(nullptr) {
        load_fvecs_aligned(path, dim_, pitch_, max_elements, n_elements_, &data_);
    }
    ~FvecsDataset() {
        if (data_) free(data_); 
    }

    size_t size() const { return n_elements_; }
    size_t dim() const { return dim_; }
    const float* get_vector(size_t index) const {
        assert(index < n_elements_);
        return data_ + index * pitch_; 
    }

private:
    size_t dim_;
    size_t pitch_;
    size_t n_elements_;
    float* data_;
};

// Dataset with metadata -> labels
class FvecsDatasetWithMeta : public FvecsDataset {
public:
    FvecsDatasetWithMeta(size_t dim, size_t max_elements,
                         const std::string& path, const std::string& meta_path)
        : FvecsDataset(dim, max_elements, path),
          meta_data_(load_bmeta(meta_path, size())) {}

    label_t get_label(size_t index) const {
        assert(index < size());
        uint32_t id = static_cast<uint32_t>(index);
        uint32_t meta = static_cast<uint32_t>(meta_data_[index]);
        return (static_cast<label_t>(meta) << 32) | id; 
    }

private:
    std::vector<meta_t> meta_data_;
};

// Generic query set template inherits from FvecsDataset
template <typename FilterT = BaseFilterFunctor>
class FvecsQueryset : public FvecsDataset {
public:
    FvecsQueryset(size_t dim, size_t max_elements,
                  const std::string& path, const std::string& meta_path)
        : FvecsDataset(dim, max_elements, path),
          filters_(load_qmeta<FilterT>(meta_path, size())) {}

    // reuse size(), dim(), get_vector() from base
    FilterT get_filter(size_t index) const {
        assert(index < size());
        return filters_[index]; 
    }

private:
    std::vector<FilterT> filters_;
};

// Aliases for specific query types
using FvecsRangeQueryset = FvecsQueryset<RangeGASFilterFunctor>;
using FvecsTagQueryset   = FvecsQueryset<TagGASFilterFunctor>;

} // namespace gaslib
