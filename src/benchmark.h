#pragma once
#include "types.h"
#include "filter.h"

namespace gaslib {

// Interface for dataset and index
template <typename T>
concept DataSetConcept = requires(const T& ds, std::size_t i) {
    { ds.size() }        -> std::convertible_to<std::size_t>;
    { ds.dim() }         -> std::convertible_to<std::size_t>;
    { ds.get_vector(i) } -> std::same_as<const float*>;
    { ds.get_label(i) }  -> std::same_as<label_t>;
};

template <typename T, typename Filter>
concept QuerySetConcept = requires(const T& qs, std::size_t i) {
    { qs.size() }        -> std::convertible_to<std::size_t>;
    { qs.dim() }         -> std::convertible_to<std::size_t>;
    { qs.get_vector(i) } -> std::same_as<const float*>;
    { qs.get_filter(i) } -> std::convertible_to<Filter>;
};

template <typename dist_t, DataSetConcept dataset_t, FilterConcept filter_t>
class IIndex {
public:
    virtual ~IIndex() = default;

    // For Benchmarking
    virtual std::string name() const = 0;

    // Build index from dataset
    virtual void build(const dataset_t& dataset) = 0;
    
    // Search for k nearest neighbors with filter
    virtual auto search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) const 
        -> std::priority_queue<std::pair<dist_t, label_t>> = 0;
    
    // Cleaning context after search (Optional, will count time)
    virtual void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) {}

    // For Statistics (Optional, do not count time)
    virtual void after_search_stat(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) {}

    // For Statistics (Optional, do not count time)
    virtual void after_search_clean() {}

    // Get statistics without clearing shortcuts (Optional)
    virtual std::vector<size_t> get_statistics() { return {}; }

    // Renew index as if it's just built (Optional)
    virtual std::vector<size_t> renew() { return {}; }

    // For Index Serialization (Optional)
    virtual bool load(const std::string& filename) { return false; }
    virtual bool save(const std::string& filename) const { return false; }
    virtual bool supports_meta_change() const { return false; }
    virtual void replace_meta(const dataset_t& dataset) {}
};


} // namespace gaslib


