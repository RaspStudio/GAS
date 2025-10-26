#pragma once
#include "types.h"
#include "benchmark.h"
#include "hnsw.h"
#include "opt.h"

namespace gaslib {

template <typename dataset_t, typename filter_t = BaseFilterFunctor>
requires DataSetConcept<dataset_t> &&
          FilterConcept<filter_t>
class GroundTruthIndex : public IIndex<float, dataset_t, filter_t> {
public:
    ihnswlib::SpaceInterface<float> *space_;
    ihnswlib::BruteforceSearch<float> *index_;

    GroundTruthIndex(dataset_t& dataset) {
#if defined(OPT_AVX512)
        space_ = new L2SpaceOptAVX512Dim16(dataset.dim());
#else
        space_ = new ihnswlib::L2Space(dataset.dim());
#endif
        index_ = new ihnswlib::BruteforceSearch<float>(space_, dataset.size());
    }

    ~GroundTruthIndex() override {
        delete index_;
        delete space_;
    }

    void build(const dataset_t& dataset) override {
        size_t n = dataset.size();
        for (size_t i = 0; i < n; ++i) {
            const float* vec = dataset.get_vector(i);
            label_t label = dataset.get_label(i);
            index_->addPoint(vec, label);
        }
    }

    std::priority_queue<std::pair<float, label_t>> search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) const override {
        return index_->searchKnn(query, k, filter);
    }

    void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {}

    std::string name() const override { return "Pre-Filter"; }
};

template <typename dataset_t, typename filter_t = BaseFilterFunctor, 
            size_t search_stretegy = 2, 
            size_t adedge_ver = 4, size_t adedge_size = 2,
            size_t shortcut_ver = 5, size_t shortcut_size = 4,
            size_t triprune_ver = 0, size_t candprune_ver = 1, size_t M = 16, size_t efc = 200>
requires DataSetConcept<dataset_t> &&
          FilterConcept<filter_t>
class GASHNSW : public IIndex<float, dataset_t, filter_t> {
    using Graph = HierarchicalNSW<float, true>;
public:
    ihnswlib::SpaceInterface<float> *space_;
    HierarchicalNSW<float, true> *index_;

    GASHNSW(dataset_t& dataset) {
#if defined(OPT_AVX512)
        space_ = new L2SpaceOptAVX512Dim16(dataset.dim());
#else
        space_ = new ihnswlib::L2Space(dataset.dim());
#endif
        index_ = new HierarchicalNSW<float, true>(space_, dataset.size(), M, efc);
    }

    ~GASHNSW() override {
        delete index_;
        delete space_;
    }

    void build(const dataset_t& dataset) override {
        size_t n = dataset.size();
        for (size_t i = 0; i < n; ++i) {
            const float* vec = dataset.get_vector(i);
            label_t label = dataset.get_label(i);
            index_->addPoint(vec, label);
        }
    }

    std::priority_queue<std::pair<float, label_t>> search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) const override {
        constexpr bool record_tree = !(search_stretegy == Graph::S_HNSW);
        return index_->searchKnn<search_stretegy, adedge_ver != 0, shortcut_ver != 0, triprune_ver != 0, candprune_ver != 0, record_tree, filter_t>(query, k, filter, ef, mode);
    }

    void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {
        constexpr bool record_tree = !(search_stretegy == Graph::S_HNSW);
        if constexpr (!record_tree) {
            // index_->clean();
        } else {
            index_->gas_clean<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>(query, k, filter, ef, mode != 0);
        }
    }

    void after_search_stat(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {
        constexpr bool record_tree = !(search_stretegy == Graph::S_HNSW);
        if constexpr (record_tree) {
            index_->gas_clean_stat<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>(query, k, filter, ef, mode != 0);
        }
    }

    void after_search_clean() override {
        index_->clean();
    }

    std::vector<size_t> renew() override {
        return index_->renew();
    }

    std::vector<size_t> get_statistics() override {
        return index_->get_statistics();
    }

    std::string name() const override {
        if constexpr (search_stretegy == Graph::S_GAS) {
            std::string s = "GAS-HNSW";
            if constexpr (M != 16) {
                s += "-M" + std::to_string(M);
            }
            if constexpr (efc != 200) {
                s += "-efc" + std::to_string(efc);
            }
            if constexpr (adedge_ver != 0) {
                s += "-AE" + std::to_string(adedge_ver) + "/" + std::to_string(adedge_size);
            }
            if constexpr (shortcut_ver != 0) {
                s += "-SC" + std::to_string(shortcut_ver) + "/" + std::to_string(shortcut_size);
            }
            if constexpr (triprune_ver != 0) {
                s += "-TP" + std::to_string(triprune_ver);
            }
            if constexpr (candprune_ver != 0) {
                s += "-CP" + std::to_string(candprune_ver);
            }
            return s;
        } else {
            return "GAS-HNSW-Unknown";
        }
    }

    std::string get_path() const {
        std::string path = "iHNSW_MAX" + std::to_string(index_->max_elements_) + "D" + std::to_string(index_->data_size_);
        if constexpr (M != 16) {
            path += "M" + std::to_string(M);
        }
        if constexpr (efc != 200) {
            path += "efc" + std::to_string(efc);
        }
        return path + ".index";
    }

    bool load(const std::string& path) override {
        const std::string& location = path + get_path();
        std::ifstream input(location, std::ios::binary);
        if (!input.is_open())
            return false;
        index_->load(location);
        // index_->load_gas(location);// TODO: fix
        return true;
    }

    bool save(const std::string& path) const override {
        // index_->save_gas(location);// TODO: fix
        return index_->save(path + get_path());
    }

    bool supports_meta_change() const override { return true; }

    void replace_meta(const dataset_t& dataset) {
        for (size_t i = 0; i < dataset.size(); ++i) {
            index_->replace_meta(i, dataset.get_label(i));
        }
    }

};

} // namespace gaslib 