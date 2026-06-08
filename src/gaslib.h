#pragma once
#include "types.h"
#include "benchmark.h"
#include "hnsw.h"
#include "hnsw_opt.h"
#include "opt.h"

namespace gaslib {

template <typename dataset_t, typename filter_t = BaseFilterFunctor>
requires DataSetConcept<dataset_t> &&
          FilterConcept<filter_t>
class PreFilterIndex : public IIndex<float, dataset_t, filter_t> {
public:
    ihnswlib::SpaceInterface<float> *space_;
    ihnswlib::BruteforceSearch<float> *index_;

    PreFilterIndex(dataset_t& dataset) {
#if defined(OPT_AVX512)
        space_ = new L2SpaceOptAVX512Dim16(dataset.dim());
#else
        space_ = new ihnswlib::L2Space(dataset.dim());
#endif
        index_ = new ihnswlib::BruteforceSearch<float>(space_, dataset.size());
    }

    ~PreFilterIndex() override {
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

    void replace_meta(const dataset_t& dataset) {
        std::vector<label_t> labels;
        labels.resize(dataset.size());
        for (size_t i = 0; i < dataset.size(); ++i) {
            labels[i] = dataset.get_label(i);
        }
        index_->replace_meta(labels);
    }

    void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {}
    std::string name() const override { return "Pre-Filter"; }
};

template <typename dataset_t, typename filter_t = BaseFilterFunctor>
requires DataSetConcept<dataset_t> &&
          FilterConcept<filter_t>
class PreFilterIndexExtend : public IIndex<float, dataset_t, filter_t> {
private:
    const MetaSet* metaset_;
public:
    ihnswlib::SpaceInterface<float> *space_;
    ihnswlib::BruteforceSearch<float> *index_;

    PreFilterIndexExtend(dataset_t& dataset, const MetaSet* metaset) : metaset_(metaset) {
#if defined(OPT_AVX512)
        space_ = new L2SpaceOptAVX512Dim16(dataset.dim());
#else
        space_ = new ihnswlib::L2Space(dataset.dim());
#endif
        index_ = new ihnswlib::BruteforceSearch<float>(space_, dataset.size());
    }

    ~PreFilterIndexExtend() override {
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
        if constexpr (!gaslib::is_composed_filter_v<filter_t>) {
            throw std::runtime_error("Only Support Composed Filter (Primary+Secondary)");
        } else {
            if (filter == nullptr) {
                throw std::runtime_error("Only Support Composed Filter (null filter)");
            }
            if (metaset_ == nullptr) {
                throw std::runtime_error("GasIndexExtend: composed filter requires MetaSet");
            }

            using PrimaryFilterT = typename gaslib::composed_filter_traits<filter_t>::primary;
            using SecondaryFilterT = typename gaslib::composed_filter_traits<filter_t>::secondary;

            auto* primary = static_cast<PrimaryFilterT*>(filter);
            const SecondaryFilterT& secondary = filter->secondary();
            auto raw = index_->searchKnn(query, 1000*k, primary);

            std::priority_queue<std::pair<float, label_t>> filtered;
            while (!raw.empty()) {
                auto item = raw.top();
                raw.pop();

                nodeid_t id = static_cast<nodeid_t>(item.second & 0xFFFFFFFFu);
                if (static_cast<size_t>(id) >= metaset_->labels.size()) continue;
                label_t label2 = metaset_->labels[static_cast<size_t>(id)];
                if (secondary(label2)) {
                    filtered.push(item);
                }
            }

            while (filtered.size() > k) {
                filtered.pop();
            }
            return filtered;
        }
    }

    void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {}

    std::string name() const override { return "Pre-Filter"; }
};


template <typename dataset_t, typename filter_t = BaseFilterFunctor, bool inline_filter = true>
requires DataSetConcept<dataset_t> &&
          FilterConcept<filter_t>
class PostFilterIndex : public IIndex<float, dataset_t, filter_t> {
public:
    ihnswlib::SpaceInterface<float> *space_;
    ihnswlib::HierarchicalNSW<float> *index_;

    std::unique_ptr<WrappedFilterFunctor> wrapped_filter;

    PostFilterIndex(dataset_t& dataset) {
        space_ = new ihnswlib::L2Space(dataset.dim());
        index_ = new ihnswlib::HierarchicalNSW<float>(space_, dataset.size());
        wrapped_filter = nullptr;
    }

    ~PostFilterIndex() override {
        delete index_;
        delete space_;
    }

    void build(const dataset_t& dataset) override {
        std::vector<label_t> labels;
        labels.resize(dataset.size());
        size_t n = dataset.size();
        for (size_t i = 0; i < n; ++i) {
            const float* vec = dataset.get_vector(i);
            label_t label = dataset.get_label(i);
            index_->addPoint(vec, label);
            labels[i] = label;
        }
        wrapped_filter = std::make_unique<WrappedFilterFunctor>(nullptr, std::move(labels));
    }

    std::priority_queue<std::pair<float, label_t>> search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) const override {
        wrapped_filter->filter_ = filter;
        if (inline_filter) {
            index_->setEf(ef);
            return index_->searchKnn(query, k, wrapped_filter.get());            
        } else {
            size_t take = std::max(k, ef) * 20;
            auto result = index_->searchKnn(query, take);
            std::priority_queue<std::pair<float, label_t>> filtered_result;

            while (!result.empty()) {
                auto item = result.top();
                result.pop();
                if ((*wrapped_filter)(item.second)) {
                    filtered_result.push(item);
                }
            }
            while (filtered_result.size() > k) {
                filtered_result.pop();
            }
            return filtered_result;
        }
    }

    void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {}

    std::vector<size_t> renew() override {
        size_t hops = index_->metric_hops;
        size_t distance_computations = index_->metric_distance_computations;
        index_->metric_hops = 0;
        index_->metric_distance_computations = 0;
        return {hops, distance_computations};
    }

    std::string name() const override { return inline_filter ? "In-Filter" : "Post-Filter"; }

    bool load(const std::string& path) override {
        const std::string& location = path + "HNSW_MAX" + std::to_string(index_->max_elements_) + "D" + std::to_string(index_->data_size_) + ".index";
        std::ifstream input(location, std::ios::binary);
        if (!input.is_open())
            return false;
        index_->loadIndex(location, space_, index_->max_elements_);
        return true;
    }

    bool save(const std::string& path) const override {
        index_->saveIndex(path + "HNSW_MAX" + std::to_string(index_->max_elements_) + "D" + std::to_string(index_->data_size_) + ".index");
        return true;
    }

    bool supports_meta_change() const override { return true; }

    void replace_meta(const dataset_t& dataset) {
        std::vector<label_t> labels;
        labels.resize(dataset.size());
        for (size_t i = 0; i < dataset.size(); ++i) {
            labels[i] = dataset.get_label(i);
        }
        if (wrapped_filter.get() == nullptr) {
            wrapped_filter = std::make_unique<WrappedFilterFunctor>(nullptr, std::move(labels));
        } else {
            wrapped_filter->labels_ = std::move(labels);
        }
    }

};


template <typename dataset_t, typename filter_t = BaseFilterFunctor,
            size_t search_stretegy = 2,
            bool continue_layout = true>
requires DataSetConcept<dataset_t> &&
          FilterConcept<filter_t>
class GasIndex : public IIndex<float, dataset_t, filter_t> {
public:
    static constexpr int S_HNSW = 0;
    static constexpr int S_BASELINE = 1;
    static constexpr int S_GAS = 2;
    static constexpr size_t adedge_ver   = (search_stretegy == S_GAS) ? 4 : 0;
    static constexpr size_t adedge_size  = (search_stretegy == S_GAS) ? 2 : 0;
    static constexpr size_t shortcut_ver = (search_stretegy == S_GAS) ? 5 : 0;
    static constexpr size_t shortcut_size = (search_stretegy == S_GAS) ? 4 : 0;
    static constexpr size_t triprune_ver = (search_stretegy == S_GAS) ? 0 : 0;
    static constexpr size_t candprune_ver = (search_stretegy == S_GAS) ? 1 : 0;
    
    static constexpr bool use_opt = continue_layout && (search_stretegy == S_GAS);
    using Graph = std::conditional_t<use_opt,
                                     HierarchicalNSWOpt<float, true>,
                                     HierarchicalNSW<float, true>>;

    ihnswlib::SpaceInterface<float> *space_;
    Graph *index_;
    size_t max_ae_;
    size_t max_sc_;
    size_t M_;
    size_t efc_;

    GasIndex(dataset_t& dataset, size_t max_additional_edges_per_node = 8, size_t max_shortcuts_per_node = 32,
             size_t M = 16, size_t efc = 200)
        : max_ae_(max_additional_edges_per_node), max_sc_(max_shortcuts_per_node),
          M_(M), efc_(efc) {

        if constexpr (search_stretegy == S_HNSW) {
            std::cout << "Creating HNSWlib Baseline Index";
        } else if constexpr (search_stretegy == S_BASELINE) {
            std::cout << "Creating HNSW Baseline Index";
        } else if constexpr (search_stretegy == S_GAS) {
            std::cout << "Creating GasIndex with max AE: " << max_additional_edges_per_node
                  << " and max SC: " << max_shortcuts_per_node;
        }
        if constexpr (use_opt) std::cout << " [continue_layout]";
        std::cout << std::endl << std::flush;
#if defined(OPT_AVX512)
        space_ = new L2SpaceOptAVX512Dim16(dataset.dim());
#else
        space_ = new ihnswlib::L2Space(dataset.dim());
#endif
        if constexpr (use_opt) {
            index_ = new Graph(space_, dataset.size(), M_, efc_, max_additional_edges_per_node, max_shortcuts_per_node);
        } else {
            index_ = new Graph(space_, dataset.size(), M_, efc_, max_additional_edges_per_node, max_shortcuts_per_node);
        }
    }

    ~GasIndex() override {
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
        constexpr bool record_tree = !(search_stretegy == S_HNSW);
        return index_->template searchKnn<search_stretegy, adedge_ver != 0, shortcut_ver != 0, triprune_ver != 0, candprune_ver != 0, record_tree, filter_t>(query, k, filter, ef, mode);
    }

    void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {
        constexpr bool record_tree = !(search_stretegy == S_HNSW);
        if constexpr (record_tree) {
            index_->template gas_clean_with_transition<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>(query, k, filter, ef, mode != 0);
        }
    }

    void status_clean() override{
        constexpr bool record_tree = !(search_stretegy == S_HNSW);
        if constexpr (search_stretegy == S_HNSW) {
            return;
        } else if constexpr (!record_tree) {
            index_->clean_with_transition();
        } else {
            index_->gas_status_clean_with_transition();
        }
    }

    void after_search_stat(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {
        constexpr bool record_tree = !(search_stretegy == S_HNSW);
        if constexpr (record_tree) {
            index_->template gas_clean_stat_with_transition<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>(query, k, filter, ef, mode != 0);
        }
    }

    std::vector<size_t> renew() override {
        return index_->renew();
    }

    std::vector<size_t> get_statistics() override {
        return index_->get_statistics();
    }

    std::string name() const override {
        if constexpr (search_stretegy == S_HNSW) {
            return "HNSWlib-Baseline";
        } else if constexpr (search_stretegy == S_BASELINE) {
            return "HNSW-Baseline";
        } else if constexpr (search_stretegy == S_GAS) {
            std::string s = "GAS";
            if constexpr (continue_layout) {
                s += "-Opt";
            }
            if (M_ != 16) {
                s += "-M" + std::to_string(M_);
            }
            if (efc_ != 200) {
                s += "-efc" + std::to_string(efc_);
            }
            s += " (AE" + std::to_string(adedge_size) + "/" + std::to_string(max_ae_)
               + ", SC" + std::to_string(shortcut_size) + "/" + std::to_string(max_sc_) + ")";
            return s;
        } else {
            return "iHNSW-Unknown";
        }
    }

    std::string get_path() const {
        std::string path = "iHNSW_MAX" + std::to_string(index_->max_elements_) + "D" + std::to_string(index_->data_size_);
        if (M_ != 16) {
            path += "M" + std::to_string(M_);
        }
        if (efc_ != 200) {
            path += "efc" + std::to_string(efc_);
        }
        return path + ".index";
    }

    bool load(const std::string& path) override {
        const std::string& location = path + get_path();
        std::ifstream input(location, std::ios::binary);
        if (!input.is_open())
            return false;
        index_->load(location);
        return true;
    }

    bool save(const std::string& path) const override {
        return index_->save(path + get_path());
    }

    bool supports_meta_change() const override { return true; }

    void replace_meta(const dataset_t& dataset) {
        for (size_t i = 0; i < dataset.size(); ++i) {
            index_->replace_meta(i, dataset.get_label(i));
        }
    }

    void replace_meta(const std::string& meta_path) {
        std::vector<meta_t> bmetas = load_bmeta(meta_path, index_->max_elements_);
        for (size_t i = 0; i < index_->max_elements_; ++i) {
            uint32_t id = static_cast<uint32_t>(i);
            uint32_t m = static_cast<uint32_t>(bmetas[i]);
            index_->replace_meta(i, (static_cast<label_t>(m) << 32) | id);
        }
        index_->clear_shortcuts();
    }
};

template <typename dataset_t, typename filter_t,
            size_t search_stretegy = 0,
            bool continue_layout = false>
requires DataSetConcept<dataset_t> &&
          FilterConcept<filter_t>
class GasIndexExtend : public GasIndex<dataset_t, filter_t,
                                            search_stretegy,
                                            continue_layout> {
    using Base = GasIndex<dataset_t, filter_t,
                            search_stretegy,
                            continue_layout>;

public:
    explicit GasIndexExtend(dataset_t& dataset, const MetaSet* metaset, size_t max_additional_edges_per_node = 8, size_t max_shortcuts_per_node = 32,
                            size_t M = 16, size_t efc = 200)
        : Base(dataset, max_additional_edges_per_node, max_shortcuts_per_node, M, efc), metaset_(metaset) {}

    bool supports_meta_change() const override {
        // MetaSet cannot be updated through replace_meta(dataset), so disable meta-change mode.
        return true;
    }

    std::priority_queue<std::pair<float, label_t>> search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) const override {
        if constexpr (!gaslib::is_composed_filter_v<filter_t>) {
            throw std::runtime_error("Only Support Composed Filter (Primary+Secondary)");
        } else {
            if (filter == nullptr) {
                throw std::runtime_error("Only Support Composed Filter (null filter)");
            }
            if (metaset_ == nullptr) {
                throw std::runtime_error("GasIndexExtend: composed filter requires MetaSet");
            }

            using PrimaryFilterT = typename gaslib::composed_filter_traits<filter_t>::primary;
            using SecondaryFilterT = typename gaslib::composed_filter_traits<filter_t>::secondary;

            auto* primary = static_cast<PrimaryFilterT*>(filter);
            const SecondaryFilterT& secondary = filter->secondary();

            constexpr bool record_tree = !(search_stretegy == Base::S_HNSW);
            const size_t take = std::max(k, ef) * 20;
            auto raw = this->index_->template searchKnn<search_stretegy,
                                                       Base::adedge_ver != 0,
                                                       Base::shortcut_ver != 0,
                                                       Base::triprune_ver != 0,
                                                       Base::candprune_ver != 0,
                                                       record_tree,
                                                       PrimaryFilterT>(query, take, primary, ef, mode);

            std::priority_queue<std::pair<float, label_t>> filtered;
            while (!raw.empty()) {
                auto item = raw.top();
                raw.pop();

                nodeid_t id = static_cast<nodeid_t>(item.second & 0xFFFFFFFFu);
                if (static_cast<size_t>(id) >= metaset_->labels.size()) continue;
                label_t label2 = metaset_->labels[static_cast<size_t>(id)];
                if (secondary(label2)) {
                    filtered.push(item);
                }
            }

            while (filtered.size() > k) {
                filtered.pop();
            }
            return filtered;
        }
    }

    void after_search(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {
        if constexpr (!gaslib::is_composed_filter_v<filter_t>) {
            throw std::runtime_error("Only Support Composed Filter (Primary+Secondary)");
        } else {
            if (filter == nullptr) {
                throw std::runtime_error("Only Support Composed Filter (null filter)");
            }
            constexpr bool record_tree = !(search_stretegy == Base::S_HNSW);
            if constexpr (record_tree) {
                // using PrimaryFilterT = typename gaslib::composed_filter_traits<filter_t>::primary;
                // auto* primary = static_cast<PrimaryFilterT*>(filter);
                this->index_->template gas_clean_with_transition<Base::adedge_ver, Base::adedge_size, Base::shortcut_ver, Base::shortcut_size, Base::triprune_ver>(query, k, filter, ef, mode != 0);
            }
        }
    }

    void after_search_stat(const float* query, size_t k, filter_t* filter, size_t ef, size_t mode = 0) override {
        if constexpr (!gaslib::is_composed_filter_v<filter_t>) {
            throw std::runtime_error("Only Support Composed Filter (Primary+Secondary)");
        } else {
            if (filter == nullptr) {
                throw std::runtime_error("Only Support Composed Filter (null filter)");
            }
            constexpr bool record_tree = !(search_stretegy == Base::S_HNSW);
            if constexpr (record_tree) {
                // using PrimaryFilterT = typename gaslib::composed_filter_traits<filter_t>::primary;
                // auto* primary = static_cast<PrimaryFilterT*>(filter);
                this->index_->template gas_clean_stat_with_transition<Base::adedge_ver, Base::adedge_size, Base::shortcut_ver, Base::shortcut_size, Base::triprune_ver>(query, k, filter, ef, mode != 0);
            }
        }
    }

private:
    const MetaSet* metaset_;
};

} // namespace gaslib 