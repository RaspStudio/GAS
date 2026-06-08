#pragma once
#include "types.h"
#include "filter.h"
#include <map>

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

    // Cleaning Status List (Optional, do not count time)
    virtual void status_clean() {}

    // Get statistics without clearing shortcuts/shortcuts (Optional)
    virtual std::vector<size_t> get_statistics() { return {}; }

    // Renew index as if it's just built (Optional)
    virtual std::vector<size_t> renew() { return {}; }

    // For Index Serialization (Optional)
    virtual bool load(const std::string& filename) { return false; }
    virtual bool save(const std::string& filename) const { return false; }
    virtual bool supports_meta_change() const { return false; }
    virtual void replace_meta(const dataset_t& dataset) {}
    virtual void replace_meta(const std::string& meta_path) {}
};

static std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    fields.push_back(cur);
    return fields;
}

std::map<size_t, std::pair<std::string, std::string>> load_query_plan(const std::string& filepath) {
    std::map<size_t, std::pair<std::string, std::string>> mp;

    std::ifstream fin(filepath);
    if (!fin.is_open()) {
        std::cerr << "❌ Failed to open query plan file: " << filepath << std::endl;
        exit(-1);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        auto fields = parse_csv_line(line);
        if (fields.size() < 3) continue;

        int idx = stoi(fields[0]);
        mp[idx] = {fields[1], fields[2]};
    }
    return mp;
}


} // namespace gaslib


