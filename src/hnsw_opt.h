#pragma once
#include "gas_storage.h"
#include "types.h"
#include "status_list.h"
#include "status_list_pool.h"
#include "hnswlib/visited_list_pool.h"
#include "hnswlib/hnswlib.h"
#include "filter.h"
#include "prio_queue.h"
#include "graph.h"
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <memory>
#include <cmath>
#include <atomic>

namespace gaslib {

template <typename dist_t, bool collect_metrics = false>
class HierarchicalNSWOpt {

    using dist_id_t = std::pair<dist_t, nodeid_t>;

    struct CompareByFirst {
        inline bool operator()(dist_id_t const& a, dist_id_t const& b) const noexcept {
            return a.first < b.first;
        }
    };

    template <typename T>
    using stdheap_t = std::priority_queue<T, std::vector<T>, CompareByFirst>;
    template <typename T>
    using iheap_t = PriorityQueue<T, CompareByFirst>;

public:
    size_t max_elements_{ 0 };
    size_t M_{ 0 };
    size_t maxM_{ 0 };
    size_t maxM0_{ 0 };
    size_t ef_construction_{ 0 };

    double mult_{ 0.0 };
    int maxlevel_{ 0 };

    std::unique_ptr<VisitedList> visited_list{nullptr};

    tableint enterpoint_node_{ 0 };

    char** linkLists_{ nullptr };

    size_t data_size_{ 0 };

    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<size_t> metric_distance_computations{ 0 };
    mutable std::atomic<size_t> metric_hops{ 0 };

    mutable std::atomic<size_t> metric_gas_prune_case1{ 0 };
    mutable std::atomic<size_t> metric_gas_prune_case2{ 0 };
    mutable std::atomic<size_t> metric_gas_prune_case3{ 0 };
    mutable std::atomic<size_t> metric_gas_prune_case4{ 0 };
    mutable std::atomic<size_t> metric_gas_prune_case5{ 0 };

    mutable std::atomic<size_t> metric_hops_upper_layer{ 0 };
    mutable std::atomic<size_t> metric_hops_approach{ 0 };
    mutable std::atomic<size_t> metric_hops_gather{ 0 };
    mutable std::atomic<size_t> metric_hops_exclude{ 0 };    
    mutable std::atomic<size_t> metric_dist_upper_layer{ 0 };
    mutable std::atomic<size_t> metric_dist_approach{ 0 };
    mutable std::atomic<size_t> metric_dist_gather{ 0 };
    mutable std::atomic<size_t> metric_dist_exclude{ 0 };
    mutable std::atomic<size_t> metric_pq_sort{ 0 };

    mutable std::atomic<size_t> metric_gas_prune_anchor_hit_{ 0 };
    mutable std::atomic<size_t> metric_gas_prune_anchor_miss_{ 0 };

    ///////////
    GASStorage storage_;
    GraphCompute<dist_t> distf_;
    GASShortcutStorage<dist_t> shortcut_storage_;
    // mutable StatusList<dist_t> status_list_;
    mutable StatusListPool<dist_t> status_list_pool_;

    StatusListPool<dist_t>* get_status_list_pool_(){
        return &status_list_pool_;
    }

#ifdef USE_SSE

    inline void _prefetch_pointer(const void* ptr) const noexcept {
        _mm_prefetch(ptr, _MM_HINT_T0);
    }

    inline void _prefetchw_pointer(const void* ptr) const noexcept {
        _mm_prefetch(ptr, _MM_HINT_ET0);
    }

    inline void _prefetch_vector(nodeid_t id) const noexcept {
        _mm_prefetch(static_cast<void*>(storage_.get_data_ptr(id)), _MM_HINT_T0);
    }

    inline void _prefetch_label(nodeid_t id) const noexcept {
        _mm_prefetch(static_cast<void*>(storage_.get_label_ptr(id)), _MM_HINT_T0);
    }

    inline void _prefetch_neighbors(nodeid_t id) const noexcept {
        _mm_prefetch(static_cast<void*>(storage_.get_neighbors_l0(id)), _MM_HINT_T0);
    }

    inline void _prefetch_status_arr(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {
        // _prefetch_pointer(status_list->ptr_for_prefetch(id));
        _prefetch_pointer(status_list->ptr_for_prefetchbits(id));
    }

    inline void _prefetchw_status_arr(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {
        _prefetchw_pointer(status_list->ptr_for_prefetch(id));
        // _prefetchw_pointer(status_list->ptr_for_prefetchbits(id));
    }

    template <bool use_gas = false>
    inline void prefetch_dist_candidate(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {
        _prefetch_vector(id);
        _prefetch_status_arr(id, status_list);
    }

    template <bool use_gas = false>
    inline void prefetch_hop(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {
        _prefetch_neighbors(id);
    }

    template <bool use_gas = false>
    inline void prefetch_entry(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {
        _prefetch_vector(id);
        _prefetch_status_arr(id, status_list);
        _prefetch_neighbors(id);
    }

#else
    
    inline void prefetch_dist_candidate(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {}
    inline void prefetch_hop(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {}
    inline void prefetch_entry(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {}

#endif


    ////////////
    HierarchicalNSWOpt(
        SpaceInterface<dist_t>* s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t max_additional_edges_per_node = 8,
        size_t max_shortcuts_per_node = 32,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : 
        storage_(max_elements, s->get_data_size(), M,
                 max_additional_edges_per_node,
                 max_shortcuts_per_node),
        distf_(s->get_dist_func(), s->get_dist_func_param()),
        shortcut_storage_(max_elements, &storage_,
                       max_additional_edges_per_node,
                       max_shortcuts_per_node),
        status_list_pool_(DEFAULT_STATUS_LIST_POOL_SIZE, max_elements) {
        max_elements_ = max_elements;
        data_size_ = s->get_data_size();
        
        if (M <= 10000) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        visited_list = std::unique_ptr<VisitedList>(new VisitedList(max_elements));
        visited_list->reset();

        // initializations for special treatment of the first node
        enterpoint_node_ = 0xFFFFFFFF; // invalid id, will be set later
        maxlevel_ = -1;

        mult_ = 1 / log(1.0 * M_);
    }

    ~HierarchicalNSWOpt() {
        visited_list.reset(nullptr);
    }

    /////////////////////////////////////////////////////
    template <bool record_tree = false>
    inline nodeid_t search_downstep(const void* data_point, nodeid_t ep_id,
        int top_level, int stop_level) const noexcept {
        assert(stop_level >= 0 && stop_level <= top_level);

        nodeid_t cur_obj = ep_id;
        dist_t cur_dist = distf_.get_distance(data_point,  storage_.get_data_ptr(cur_obj));

        if constexpr (collect_metrics) {
            metric_distance_computations++;
            if constexpr (record_tree) {
                metric_dist_upper_layer++;
            }
        }
        

        for (int level = top_level; level > stop_level; level--) {
            bool changed = true;
            while (changed) {
                changed = false;

                nodeid_t* data = level ? (nodeid_t*) storage_.get_neighbors_l1(cur_obj, level) : (nodeid_t*) storage_.get_neighbors_l0(cur_obj);
                nodeid_t size = *data;
                
#ifdef USE_SSE
                
                _mm_prefetch(storage_.get_data_ptr(*(data + 1)), _MM_HINT_T0);
#endif
                for (nodeid_t i = 1; i <= size; i++) {
                    nodeid_t cand = *(data + i);
                    assert (cand < max_elements_);
#ifdef USE_SSE
                    if (i + 1 <= size) [[likely]] _mm_prefetch(storage_.get_data_ptr(*(data + i + 1)), _MM_HINT_T0);
#endif
                    dist_t d = distf_.get_distance(data_point, storage_.get_data_ptr(cand));
                    
                    if constexpr (collect_metrics) {
                        metric_distance_computations++;
                        if constexpr (record_tree) {
                            metric_dist_upper_layer++;
                        }
                    }
                    

                    if (d < cur_dist) [[unlikely]] {
                        cur_dist = d;
                        cur_obj = cand;
                        changed = true;
                    }
                }
                if constexpr (collect_metrics) {
                    metric_hops++;
                    if constexpr (record_tree) {
                        metric_hops_upper_layer++;
                    }
                }
            }
        }
        return cur_obj;
    }
    /////////////////////////////////////////////////////

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label = *storage_.get_label_ptr(internal_id);
        return return_label;
    }

    inline char* getDataByInternalId(tableint internal_id) const {
        return storage_.get_data_ptr(internal_id);
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    
    template <bool bare_bone_search = true, bool collect = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        search_base_layer(
            tableint ep_id,
            const void* data_point,
            size_t ef,
            BaseFilterFunctor* IdFilter = nullptr,
            int level = 0
            ) const {

        visited_list->reset();
        vl_type* visited_array = visited_list->mass;
        vl_type visited_array_tag = visited_list->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || ((!IdFilter) || (*IdFilter)(getExternalLabel(ep_id)))) {

            dist_t dist = distf_.get_distance(data_point, getDataByInternalId(ep_id));
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);

            if constexpr (collect_metrics && collect) {
                metric_distance_computations++;
            }
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

                visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int* data;
            if (level == 0){
                data = (int*) get_linklist0(current_node_id);
            } else{
                data = (int*) get_linklist(current_node_id, level);
            }
            size_t size = getListCount((linklistsizeint*) data);

            if constexpr (collect_metrics && collect) {
                metric_hops++;
            }

#ifdef USE_SSE
            _mm_prefetch((char*) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(storage_.get_data_ptr(*(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char*) (data + 2), _MM_HINT_T0);
#endif // LINK | DATA  | LABEL

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                //                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char*) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(storage_.get_data_ptr(*(data + j + 1)), _MM_HINT_T0); ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    
                    char* currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = distf_.get_distance(data_point, currObj1);

                    if constexpr (collect_metrics && collect) {
                        metric_distance_computations += 1;
                    }

                    bool flag_consider_candidate;
                    flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(storage_.get_neighbors_l0(candidate_set.top().second), _MM_HINT_T0); // ?
#endif

                        if (bare_bone_search || ((!IdFilter) || (*IdFilter)(getExternalLabel(candidate_id)))) {
                            top_candidates.emplace(dist, candidate_id);
                        }

                        bool flag_remove_extra = false;
                        flag_remove_extra = top_candidates.size() > ef;
                        while (flag_remove_extra) {
                            // tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            flag_remove_extra = top_candidates.size() > ef;
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        
        return top_candidates;
    }

    
    template <typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>    
    struct SearchCtx {
        
        const void*                   q;    // query data_point
        heap_t&                       res_heap;    // result_heap
        stdheap_t<dist_id_t>&         cand_heap;    // candidate_heap
        dist_t&                       res_lb;  // result_lowerbound
        size_t                        ef;   // ef
        const filter_t&               filter;    // filter
        StatusList<dist_t>*           status_list;    // status_list

        
        bool*                         k_lb_upd = nullptr;
        dist_t*                       k_lb = nullptr;       
        size_t                        k = 0;   
        std::vector<nodeid_t>* shortcut_candidates = nullptr; 
        std::unordered_set<nodeid_t>* suspended_shortcut_nodes = nullptr; 
    };

    
    template <typename filter_t>
    requires FilterConcept<filter_t>
    inline auto satisfy_filter(const nodeid_t& id, const filter_t& IdFilter) const noexcept -> bool {
        return IdFilter(getExternalLabel(id));
    }

    template <typename heap_like_t>
    inline static auto should_stop(const heap_like_t& results, dist_t cand_dist, dist_t results_lowerbound, size_t ef) noexcept -> bool {
        return (cand_dist > results_lowerbound && results.size() == ef);
    }

    template <typename heap_like_t>
    inline static auto should_consider(const heap_like_t& results, dist_t cand_dist, dist_t results_lowerbound, size_t ef) noexcept -> bool {
        return (results_lowerbound > cand_dist || results.size() < ef);
    }

    inline auto get_distance(const void* data_point, nodeid_t id) const noexcept -> dist_t {
        if constexpr (collect_metrics) metric_distance_computations++;
        return distf_.get_distance(data_point, storage_.get_data_ptr(id));
    }

    inline auto get_distance(const void* a, const void* b) const noexcept -> dist_t {
        return distf_.get_distance(a, b);
    }

    
    
    template <bool record_tree = false, bool from_shortcut = false>
    inline auto set_visited_and_disted(nodeid_t id, nodeid_t from_id, nodeid_t nhop, dist_t dist, StatusList<dist_t>* status_list) const noexcept {
        
        constexpr status_container_t flag = from_shortcut ? STATUS_VISITED | STATUS_DISTED | STATUS_FROM_SHORTCUT : STATUS_VISITED | STATUS_DISTED;
        
        status_list->set_visitbit(id);
        status_list->set_status(id, flag);
        if constexpr (record_tree) {
            status_list->set_from(id, from_id, nhop);
            status_list->set_dist(id, dist);
        }
    }

    template <bool record_tree = false>
    inline auto set_visited_no_dist(nodeid_t id, nodeid_t from_id, nodeid_t nhop, StatusList<dist_t>* status_list) const noexcept {
        
        status_list->set_visitbit(id);
        status_list->set_status(id, STATUS_VISITED);
        if constexpr (record_tree) {
            status_list->set_from(id, from_id, nhop);
        }
    }

    inline bool is_visited(nodeid_t id, StatusList<dist_t>* status_list) const noexcept {
        // return status_list->has_status(id, STATUS_VISITED);
        return status_list->get_visitbit(id);
    }

    inline bool is_visited_by(nodeid_t id, nodeid_t from_id, StatusList<dist_t>* status_list) const noexcept {
        return status_list->get_from(id) == from_id;
    }

    

    
    template <bool record_tree, typename heap_t>
    requires HeapConcept<heap_t, dist_id_t>
    inline bool fit_in_heap(heap_t& result_heap, dist_t dist,
                            nodeid_t candidate_id, size_t k, size_t ef, dist_t& result_lowerbound, StatusList<dist_t>* status_list) const noexcept {
        result_heap.emplace(dist, candidate_id);
        if constexpr (record_tree) status_list->push_gas_results(candidate_id);
        if (result_heap.size() > ef) {
            result_heap.pop();
        }
        if (!result_heap.empty()) [[likely]]
            result_lowerbound = result_heap.top().first;
        return false;
    }


    template <bool record_tree>
    inline bool fit_in_heap(iheap_t<dist_id_t>& result_heap, dist_t dist,
                            nodeid_t candidate_id, size_t k, size_t ef, dist_t& result_lowerbound,
                            bool* update, dist_t* klb, StatusList<dist_t>* status_list) const noexcept {
        result_heap.emplace(dist, candidate_id);
        if constexpr (record_tree) status_list->push_gas_results(candidate_id);
        bool new_topk = false;
        if (result_heap.size() > k) {
            
            if (klb == nullptr || update == nullptr) [[unlikely]] { 
                std::cerr << "Error: klb or update pointer is null." << std::endl; 
                exit(EXIT_FAILURE); 
            }
            if (dist < *klb) {
                
                new_topk = true;
                if (!*update) {
                    *update = true;
                }
            }
        }
        if (result_heap.size() > ef) {
            result_heap.pop();
        }
        if (!result_heap.empty()) [[likely]]
            result_lowerbound = result_heap.top().first;
        return new_topk;
    }


    template <bool record_tree = false, bool enable_shortcuts = false, typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline auto handle_entry(nodeid_t ep_id, SearchCtx<heap_t, filter_t>& ctx) const noexcept -> void {
        if constexpr (record_tree) {
            if (ctx.status_list->l0_ep_ != INVALID_NODE_ID)
                throw std::runtime_error("Error: Entry point already set in status list.");
            ctx.status_list->l0_ep_ = ep_id;
        }
        if (satisfy_filter(ep_id, ctx.filter)) {
            dist_t dist = get_distance(ctx.q, ep_id);
            ctx.res_lb = dist;
            ctx.res_heap.emplace(dist, ep_id);
            ctx.cand_heap.emplace(-dist, ep_id);
            set_visited_and_disted<record_tree>(ep_id, ep_id, 1, dist, ctx.status_list);
            ctx.status_list->push_gas_results(ep_id);
            
            // Visit shortcuts of the entry point if enabled
            if constexpr (enable_shortcuts) {
                ctx.shortcut_candidates->emplace_back(ep_id);
            }
        } else {
            ctx.res_lb = std::numeric_limits<dist_t>::max();
            ctx.cand_heap.emplace(-ctx.res_lb, ep_id);
            set_visited_no_dist<record_tree>(ep_id, ep_id, 1, ctx.status_list);
        }
    }


    
    template <bool record_tree, bool enable_shortcuts = false, bool enable_cand_pruning = false, typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline bool handle_valid_node(nodeid_t from_id, nodeid_t cur_hop, nodeid_t node_id,
                                SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        
        _prefetchw_status_arr(node_id, ctx.status_list);
        dist_t dist = get_distance(ctx.q, node_id);
        set_visited_and_disted<record_tree>(node_id, from_id, cur_hop, dist, ctx.status_list);
        
        bool updated = false;
        if (should_consider(ctx.res_heap, dist, ctx.res_lb, ctx.ef)) {
            ctx.cand_heap.emplace(-dist, node_id);
            if constexpr (enable_cand_pruning && std::same_as<heap_t, iheap_t<dist_id_t>>)
                updated = fit_in_heap<record_tree>(ctx.res_heap, dist, node_id, ctx.k, ctx.ef, ctx.res_lb, ctx.k_lb_upd, ctx.k_lb, ctx.status_list);
            else
                updated = fit_in_heap<record_tree>(ctx.res_heap, dist, node_id, ctx.k, ctx.ef, ctx.res_lb, ctx.status_list);

            if constexpr (enable_shortcuts) {
                
                assert(ctx.shortcut_candidates);
                ctx.shortcut_candidates->emplace_back(node_id);
            }
        }
        return updated;
    }


    
    template <bool record_tree, bool enable_cand_pruning = false, typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline void handle_valid_shortcut(nodeid_t from_id, nodeid_t cur_hop, nodeid_t node_id, 
                                     dist_t shortcut_start_dist, dist_t node_dist,
                                     SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        
        _prefetchw_status_arr(node_id, ctx.status_list);
        set_visited_and_disted<record_tree, true>(node_id, from_id, cur_hop, node_dist, ctx.status_list);
        
        
        if (should_consider(ctx.res_heap, node_dist, ctx.res_lb, ctx.ef)) {
            ctx.cand_heap.emplace(-node_dist, node_id);
            if constexpr (enable_cand_pruning && std::same_as<heap_t, iheap_t<dist_id_t>>)
                fit_in_heap<record_tree>(ctx.res_heap, node_dist, node_id, ctx.k, ctx.ef, ctx.res_lb, ctx.k_lb_upd, ctx.k_lb, ctx.status_list);
            else
                fit_in_heap<record_tree>(ctx.res_heap, node_dist, node_id, ctx.k, ctx.ef, ctx.res_lb, ctx.status_list);

            
            if (node_dist < shortcut_start_dist) {
                
                assert(ctx.shortcut_candidates);
                ctx.shortcut_candidates->emplace_back(node_id);
            } else {
                
                assert(ctx.suspended_shortcut_nodes);
                ctx.suspended_shortcut_nodes->insert(node_id);
            }
        }
    }


    
    template <bool record_tree = false, bool enable_shortcuts = false, bool enable_cand_pruning = false, bool revisit = false,
            typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline bool visit_neighbor(nodeid_t cur_id, nodeid_t cur_hop, nodeid_t nbr_id,
                            SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        if constexpr (revisit) {
            
            if (!is_visited_by(nbr_id, cur_id, ctx.status_list)) return false;
        } else if (is_visited(nbr_id, ctx.status_list)) return false;
        
        if constexpr (enable_cand_pruning) {
            bool updated = false;
            if (satisfy_filter(nbr_id, ctx.filter)) {
                updated = handle_valid_node<record_tree, enable_shortcuts, enable_cand_pruning>(cur_id, cur_hop, nbr_id, ctx);
            }
            return updated;
        }

        dist_t dist = get_distance(ctx.q, nbr_id);
        set_visited_and_disted<record_tree>(nbr_id, cur_id, cur_hop, dist, ctx.status_list);

        if (should_consider(ctx.res_heap, dist, ctx.res_lb, ctx.ef)) {
            ctx.cand_heap.emplace(-dist, nbr_id);

            if (satisfy_filter(nbr_id, ctx.filter)) {
                if constexpr (enable_cand_pruning && std::same_as<heap_t, iheap_t<dist_id_t>>)
                    fit_in_heap<record_tree>(ctx.res_heap, dist, nbr_id, ctx.k, ctx.ef, ctx.res_lb, ctx.k_lb_upd, ctx.k_lb, ctx.status_list);
                else
                    fit_in_heap<record_tree>(ctx.res_heap, dist, nbr_id, ctx.k, ctx.ef, ctx.res_lb, ctx.status_list);
                if constexpr (enable_shortcuts) {
                    
                    assert(ctx.shortcut_candidates);
                    ctx.shortcut_candidates->emplace_back(nbr_id);
                }
            }
        }
        return false;
    }


    
    template <bool record_tree = false, bool enable_shortcuts = false,
            bool enable_cand_pruning = false,
            typename heap_t, typename filter_t>
    inline void visit_neighbors_from_ptr(nodeid_t cur_id, nodeid_t cur_hop,
                                const nodeid_t* data, const nodeid_t size,
                                SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        prefetch_dist_candidate(*(data), ctx.status_list);

        nodeid_t updated_at = size; 

        nodeid_t j = 0;
        for (; j + 1 < size; j++) {
            nodeid_t candidate_id = *(data + j);
            prefetch_dist_candidate(*(data + j + 1), ctx.status_list);
            if (updated_at == size) {
                bool updated = visit_neighbor<record_tree, enable_shortcuts, enable_cand_pruning>(cur_id, cur_hop, candidate_id, ctx);
                if (updated) updated_at = j;
            } else {
                visit_neighbor<record_tree, enable_shortcuts, false>(cur_id, cur_hop, candidate_id, ctx);
            }
        }

        
        assert(j + 1 == size && size != 0);
        {
            nodeid_t candidate_id = *(data + j);
            prefetch_hop(ctx.cand_heap.top().second, ctx.status_list);
            if (updated_at == size) {
                bool updated = visit_neighbor<record_tree, enable_shortcuts, enable_cand_pruning>(cur_id, cur_hop, candidate_id, ctx);
                if (updated) updated_at = j;
            } else {
                visit_neighbor<record_tree, enable_shortcuts, false>(cur_id, cur_hop, candidate_id, ctx);
            }            
        }

        if (updated_at == size) return;

        
        for (j = 0; j < updated_at; j++) {
            nodeid_t candidate_id = *(data + j);
            prefetch_dist_candidate(*(data + j + 1), ctx.status_list);
            
            visit_neighbor<record_tree, enable_shortcuts, false, true>(cur_id, cur_hop, candidate_id, ctx);
        }
    }

    
    template <bool record_tree = false, bool enable_shortcuts = false,
            bool enable_cand_pruning = false,
            typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline void visit_neighbors(nodeid_t cur_id, nodeid_t cur_hop,
                                SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        auto span = storage_.get_l0_span(cur_id);
        if (span.empty()) return;
        visit_neighbors_from_ptr<record_tree, enable_shortcuts, enable_cand_pruning>(cur_id, cur_hop, span.data(), span.size(), ctx);
    }

    
    template <bool record_tree = false, bool enable_shortcuts = false,
            bool enable_cand_pruning = false,
            typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline void visit_additional_neighbors(nodeid_t cur_id, nodeid_t cur_hop,
                                        SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        const nodeid_t* data = shortcut_storage_.additional_edges().data(cur_id);
        nodeid_t size = shortcut_storage_.additional_edges().size(cur_id);
        if (size == 0) return;
        visit_neighbors_from_ptr<record_tree, enable_shortcuts, enable_cand_pruning>(cur_id, cur_hop, data, size, ctx);
    }
    
    
    template <bool record_tree = false, bool enable_cand_pruning = false, typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline void visit_shortcuts(nodeid_t cur_hop, nodeid_t nbr_id,
                                SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        size_t sc_size = shortcut_storage_.shortcuts().size(nbr_id);
        if (sc_size == 0) return;

        
        dist_t nbr_dist = ctx.status_list->get_dist(nbr_id);

        
        const RefShortcutList<nodeid_t>::Entry* sc_data = shortcut_storage_.shortcuts().data(nbr_id);

        
        prefetch_dist_candidate((sc_data)->value, ctx.status_list);

        for (nodeid_t i = 0; i + 1 < sc_size; i++) {
            nodeid_t sc_nbr_id = (sc_data + i)->value;
            prefetch_dist_candidate((sc_data + i + 1)->value, ctx.status_list);

            if (!is_visited(sc_nbr_id, ctx.status_list) && satisfy_filter(sc_nbr_id, ctx.filter)) {
                dist_t sc_nbr_dist = get_distance(ctx.q, sc_nbr_id);
                handle_valid_shortcut<record_tree, enable_cand_pruning>(nbr_id, cur_hop, sc_nbr_id, nbr_dist, sc_nbr_dist, ctx);
            }
        }

        
        {
            nodeid_t sc_nbr_id = (sc_data + sc_size - 1)->value;

            if (!is_visited(sc_nbr_id, ctx.status_list) && satisfy_filter(sc_nbr_id, ctx.filter)) {
                dist_t sc_nbr_dist = get_distance(ctx.q, sc_nbr_id);
                handle_valid_shortcut<record_tree, enable_cand_pruning>(nbr_id, cur_hop, sc_nbr_id, nbr_dist, sc_nbr_dist, ctx);
            }
        }
    }


    
    template <bool record_tree = false, bool enable_cand_pruning = false, typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    inline void visit_shortcuts_opt(nodeid_t cur_hop, nodeid_t nbr_id,
                                SearchCtx<heap_t, filter_t>& ctx) const noexcept {
        auto* sc_nl = storage_.get_sc_l0(nbr_id);
        nodeid_t sc_size = sc_nl->len_;
        if (sc_size == 0) return;

        dist_t nbr_dist = ctx.status_list->get_dist(nbr_id);
        const nodeid_t* data = sc_nl->neighbor_;

        prefetch_dist_candidate(*(data), ctx.status_list);

        nodeid_t j = 0;
        for (; j + 1 < sc_size; j++) {
            nodeid_t sc_nbr_id = *(data + j);
            prefetch_dist_candidate(*(data + j + 1), ctx.status_list);

            if (!is_visited(sc_nbr_id, ctx.status_list) && satisfy_filter(sc_nbr_id, ctx.filter)) {
                dist_t sc_nbr_dist = get_distance(ctx.q, sc_nbr_id);
                handle_valid_shortcut<record_tree, enable_cand_pruning>(nbr_id, cur_hop, sc_nbr_id, nbr_dist, sc_nbr_dist, ctx);
            }
        }

        assert(j + 1 == sc_size && sc_size != 0);
        {
            nodeid_t sc_nbr_id = *(data + j);
            prefetch_hop(ctx.cand_heap.top().second, ctx.status_list);

            if (!is_visited(sc_nbr_id, ctx.status_list) && satisfy_filter(sc_nbr_id, ctx.filter)) {
                dist_t sc_nbr_dist = get_distance(ctx.q, sc_nbr_id);
                handle_valid_shortcut<record_tree, enable_cand_pruning>(nbr_id, cur_hop, sc_nbr_id, nbr_dist, sc_nbr_dist, ctx);
            }
        }
    }


    template <bool record_tree = false, typename filter_t>
    requires FilterConcept<filter_t>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        baseline_search(
            tableint ep_id,
            const void* data_point,
            size_t ef,
            filter_t& IdFilter,
            StatusList<dist_t>* status_list) const {

        prefetch_entry(ep_id, status_list);

        if (status_list->is_dirty()) {
            throw std::runtime_error("Status list is dirty, please clear it before searching.");
        }
        status_list->set_dirty();

        stdheap_t<dist_id_t> result_heap;
        stdheap_t<dist_id_t> candidate_heap;

        dist_t result_lowerbound = std::numeric_limits<dist_t>::max();
        SearchCtx<stdheap_t<dist_id_t>, filter_t> ctx{
            /* q        */ data_point,
            /* res_heap        */ result_heap,
            /* cand_heap        */ candidate_heap,
            /* res_lb      */ result_lowerbound,
            /* ef       */ ef,
            /* filter        */ IdFilter,
            /* status_list */ status_list
        };

        handle_entry<record_tree>(ep_id, ctx);

        nodeid_t current_hop = 0;
        while (!candidate_heap.empty()) {
            current_hop++;

            dist_id_t cur_pair = candidate_heap.top();
            dist_t candidate_dist = -cur_pair.first;
            nodeid_t cur_id = cur_pair.second;

            if (should_stop(result_heap, candidate_dist, result_lowerbound, ef)) break;
            candidate_heap.pop();

            visit_neighbors<record_tree>(cur_id, current_hop, ctx);
        }

        if constexpr (collect_metrics) { 
            metric_hops += current_hop;
            status_list->max_hops_ = current_hop;
        } 

        return result_heap;
    }

    template <bool record_tree, bool enable_adedge, bool enable_shortcuts,
            bool enable_cand_pruning,
            typename heap_t, typename filter_t>
    requires FilterConcept<filter_t> && HeapConcept<heap_t, dist_id_t>
    stdheap_t<dist_id_t> gas_search(
        tableint ep_id, const void* data_point, size_t k, size_t ef, const filter_t& IdFilter, StatusList<dist_t>* status_list) const {
        assert(status_list != nullptr);

        prefetch_entry(ep_id, status_list);
        if (status_list->is_dirty()) throw std::runtime_error("Status list is dirty, please clear it before searching.");
        if (record_tree) status_list->set_gasable(ef);
        status_list->set_dirty();

        heap_t result_heap;
        stdheap_t<dist_id_t> candidate_heap;
                
        if constexpr (std::same_as<heap_t, iheap_t<dist_id_t>>) {
            result_heap.reserve(ef);
        }

        bool   k_lowerbound_need_update = true;
        dist_t k_lowerbound = std::numeric_limits<dist_t>::max();
        dist_t result_lowerbound = std::numeric_limits<dist_t>::max();

        std::vector<nodeid_t> shortcut_candidates;
        
        std::unordered_set<nodeid_t> suspended_shortcut_nodes;

        SearchCtx<heap_t, filter_t> ctx{
            /* q        */ data_point,
            /* res_heap        */ result_heap,
            /* cand_heap        */ candidate_heap,
            /* res_lb      */ result_lowerbound,
            /* ef       */ ef,
            /* filter        */ IdFilter,
            /* status_list */ status_list,
            /* k_lb_upd */ enable_cand_pruning ? &k_lowerbound_need_update : nullptr,
            /* k_lb     */ enable_cand_pruning ? &k_lowerbound : nullptr,
            /* k       */  k,
            /* shortcut_candidates */ enable_shortcuts ? &shortcut_candidates : nullptr,
            /* suspended_shortcut_nodes */ enable_shortcuts ? &suspended_shortcut_nodes : nullptr
        };

        handle_entry<record_tree, enable_shortcuts>(ep_id, ctx);
        
        nodeid_t cur_hop = 0;
        while (!candidate_heap.empty()) {
            dist_id_t cur_pair = candidate_heap.top();
            dist_t cur_dist = -cur_pair.first;
            nodeid_t cur_id = cur_pair.second;

            if (should_stop(result_heap, cur_dist, result_lowerbound, ef)) [[unlikely]] break;
            candidate_heap.pop();

            status_list->set_status(cur_id, STATUS_FROM_SHORTCUT);
            cur_hop++;

            bool prune_candidate = cur_dist > k_lowerbound;
            if constexpr (enable_cand_pruning && std::same_as<heap_t, iheap_t<dist_id_t>>) {
                if (!prune_candidate && k_lowerbound_need_update && result_heap.size() >= ef && cur_hop > k) {
                    metric_pq_sort++;
                    k_lowerbound = result_heap.sort(k).first;
                    // result_heap.sort();
                    // k_lowerbound = result_heap.data_.at(result_heap.data_.size() - k).first;
                    //                 prune_candidate = cur_dist > k_lowerbound; 
                    k_lowerbound_need_update = false;
                }
            }

            if (enable_cand_pruning && prune_candidate) {
                if constexpr (enable_adedge) {
                    auto* nl = storage_.get_neighbors_l0(cur_id);
                    visit_neighbors_from_ptr<record_tree, enable_shortcuts, true>(cur_id, cur_hop, nl->neighbor_, nl->len_, ctx);
                } else {
                    visit_neighbors<record_tree, enable_shortcuts, true>(cur_id, cur_hop, ctx);
                }
            } else {
                if constexpr (enable_adedge) {
                    auto* nl = storage_.get_neighbors_l0(cur_id);
                    visit_neighbors_from_ptr<record_tree, enable_shortcuts, false>(cur_id, cur_hop, nl->neighbor_, nl->len_, ctx);
                } else {
                    visit_neighbors<record_tree, enable_shortcuts, false>(cur_id, cur_hop, ctx);
                }
            }

            if constexpr (enable_shortcuts) {
                assert(ctx.suspended_shortcut_nodes);
                assert(ctx.shortcut_candidates);
                if (ctx.suspended_shortcut_nodes->count(cur_id)) {
                    ctx.suspended_shortcut_nodes->erase(cur_id);
                    visit_shortcuts_opt<record_tree, enable_cand_pruning>(cur_hop, cur_id, ctx);
                }
                while (!ctx.shortcut_candidates->empty()) {
                    nodeid_t nbr_id = ctx.shortcut_candidates->back();
                    ctx.shortcut_candidates->pop_back();
                    visit_shortcuts_opt<record_tree, enable_cand_pruning>(cur_hop, nbr_id, ctx);
                }
            }
        }

        if constexpr (enable_shortcuts) {
            assert(ctx.shortcut_candidates);
            while (!ctx.shortcut_candidates->empty()) {
                nodeid_t nbr_id = ctx.shortcut_candidates->back();
                ctx.shortcut_candidates->pop_back();
                visit_shortcuts_opt<record_tree, enable_cand_pruning>(cur_hop, nbr_id, ctx);
            }
        }

        if constexpr (collect_metrics) { metric_hops += cur_hop; status_list->max_hops_ = cur_hop; }

        if constexpr (std::same_as<heap_t, iheap_t<dist_id_t>>) {
            stdheap_t<dist_id_t> final_heap(CompareByFirst(), std::move(result_heap.data_));
            return final_heap;
        } else {
            return result_heap;
        }
    }


    template <GasFilterConcept filter_t>
    auto bruteforce_search(
            const void* data_point,
            size_t k,
            filter_t& IdFilter) const -> stdheap_t<dist_id_t> {
        stdheap_t<dist_id_t> result_heap;

        for (size_t i = 0; i < storage_.cur_elements_; i++) {
            if (satisfy_filter(i, IdFilter)) {
                dist_t dist = get_distance(data_point, storage_.get_data_ptr(i));
                result_heap.emplace(dist, i);
            }

            if (result_heap.size() > k) {
                result_heap.pop();
            }
        }

        return result_heap;
    }

    auto bruteforce_search(
            const void* data_point,
            size_t k) const -> stdheap_t<dist_id_t> {
        stdheap_t<dist_id_t> result_heap;

        for (size_t i = 0; i < storage_.cur_elements_; i++) {
            dist_t dist = get_distance(data_point, storage_.get_data_ptr(i));
            result_heap.emplace(dist, i);

            if (result_heap.size() > k) {
                result_heap.pop();
            }
        }

        return result_heap;
    }

    /////////////////// END SEARCH ///////////////////////////////////////////////////////////



    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>& top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                    distf_.get_distance(getDataByInternalId(second_pair.second),
                        getDataByInternalId(curent_pair.second));
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    linklistsizeint* get_linklist0(tableint internal_id) const {
        return (linklistsizeint*) storage_.get_neighbors_l0(internal_id);
    }

    linklistsizeint* get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint*) storage_.get_neighbors_l1(internal_id, level);
    }

    linklistsizeint* get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    tableint mutuallyConnectNewElement(
        const void* data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>& top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            linklistsizeint* ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint* data = (tableint*) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > storage_.level(selectedNeighbors[idx]))
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
            // Sync original_neighbor_count_ for the new element at level 0
            if (level == 0)
                storage_.init_original_neighbor_count(cur_c, selectedNeighbors.size());
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

            linklistsizeint* ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > storage_.level(selectedNeighbors[idx]))
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint* data = (tableint*) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = distf_.get_distance(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]));
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            distf_.get_distance(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx])),
                            data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = distf_.get_distance(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]));
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
            // Sync original_neighbor_count_ for existing neighbors at level 0
            if (level == 0)
                storage_.init_original_neighbor_count(selectedNeighbors[idx], getListCount(ll_other));
        }

        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) {
        storage_.resize(new_max_elements);
        visited_list.reset(new VisitedList(new_max_elements));
        max_elements_ = new_max_elements;
    }

    unsigned short int getListCount(linklistsizeint* ptr) const {
        return *((unsigned short int*)ptr);
    }

    void setListCount(linklistsizeint* ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr)) = *((unsigned short int*) & size);
    }

    /*
     * Adds point. Updates the point if it is already in the index.
     * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
     */
    void addPoint(const void* data_point, labeltype label, int level = -1) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                throw std::runtime_error("Element with the same label already exists");
            }

            label_lookup_[label] = cur_c;
        }

        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;
        cur_c = storage_.init_storage(curlevel);

        int maxlevelcopy = maxlevel_;

        tableint currObj = enterpoint_node_;
        // tableint enterpoint_copy = enterpoint_node_; // originally for thread safety

        

        // Initialisation of the data and label
        memcpy(storage_.get_label_ptr(cur_c), &label, sizeof(labeltype));
        memcpy(storage_.get_data_ptr(cur_c), data_point, data_size_);

        if ((signed) currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = distf_.get_distance(data_point, getDataByInternalId(currObj));
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int* data;
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint* datal = (tableint*) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = distf_.get_distance(data_point, getDataByInternalId(cand));
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0) // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = search_base_layer<true>(
                    currObj, data_point, ef_construction_, nullptr, level);
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
    }

    constexpr static int S_HNSW = 0;
    constexpr static int S_BASELINE = 1;
    constexpr static int S_GAS = 2;

    template<size_t search_type, bool enable_adedge, bool enable_shortcuts, bool enable_pruning, bool enable_cand_pruning, bool record_tree = false, GasFilterConcept filter_t>
    std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void* query_data, size_t k, filter_t* IdFilter, size_t ef, size_t ef_lb) const {
        
        // Acquire status_list from CLEARED pool
        StatusList<dist_t>* status_list = status_list_pool_.acquire(StatusListPool<dist_t>::State::CLEARED);
        
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (enterpoint_node_ == 0xFFFFFFFF) {
            status_list_pool_.release(status_list, StatusListPool<dist_t>::State::CLEARED);
            return result;
        }

        tableint baselayer_ep = search_downstep<record_tree>(query_data, enterpoint_node_, maxlevel_, 0);

        stdheap_t<dist_id_t> top_candidates;
        bool bare_bone_search = !IdFilter;
        if (!bare_bone_search) {
            if constexpr (search_type == S_HNSW) {
                top_candidates = search_base_layer<false, true>(
                    baselayer_ep, query_data, std::max(ef, k), static_cast<BaseFilterFunctor*>(IdFilter));
            } else if constexpr (search_type == S_BASELINE) {
                top_candidates = baseline_search<record_tree>(
                    baselayer_ep, query_data, std::max(ef, k), *IdFilter, status_list);
            } else if constexpr (search_type == S_GAS) {
                top_candidates = gas_search<record_tree, enable_adedge, enable_shortcuts, enable_cand_pruning, iheap_t<dist_id_t>>(
                    baselayer_ep, query_data, ef_lb == 0 ? k : ef_lb, std::max(k, ef), *IdFilter, status_list);
            } else {
                throw std::runtime_error("Unsupported optimization_applied value");
            }
        } else {
            status_list_pool_.release(status_list, StatusListPool<dist_t>::State::CLEARED);
            throw std::runtime_error("Empty Filter");
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        
        // Transition status_list to SEARCHED_UNCOUNTED state after search
        if constexpr (search_type != S_HNSW) {
            status_list_pool_.release(status_list, StatusListPool<dist_t>::State::SEARCHED_UNCOUNTED);
        } else {
            status_list_pool_.release(status_list, StatusListPool<dist_t>::State::CLEARED);
        }
        return result;
    }

    void clean(StatusList<dist_t>* status_list_) const {
        status_list_->template clear_all<true>();
    }

    // State-based wrappers for after_search_stat
    template <size_t adedge_ver = 0, size_t adedge_size = 0, size_t shortcut_ver = 0, size_t shortcut_size = 0, size_t triprune_ver = 0, typename filter_t>
    void gas_clean_stat_with_transition(const float* query, size_t k, filter_t* filter, size_t ef, bool use_bf = false) {
        // Acquire from SEARCHED_UNCOUNTED state
        StatusList<dist_t>* status_list = status_list_pool_.acquire(StatusListPool<dist_t>::State::SEARCHED_UNCOUNTED);
        
        // Do statistics
        gas_clean_stat<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>(status_list, query, k, filter, ef, use_bf);
        
        // Transition to SEARCHED_COUNTED state
        status_list_pool_.release(status_list, StatusListPool<dist_t>::State::SEARCHED_COUNTED);
    }

    template <size_t adedge_ver = 0, size_t adedge_size = 0, size_t shortcut_ver = 0, size_t shortcut_size = 0, size_t triprune_ver = 0, typename filter_t>
    void gas_clean_stat(StatusList<dist_t>* status_list_, const float* query, size_t k, filter_t* filter, size_t ef, bool use_bf = false) {
        std::vector<size_t> stat = status_list_->stat_stages(k);
        metric_hops_approach += stat[0];
        metric_dist_approach += stat[1];
        metric_hops_gather += stat[2];
        metric_dist_gather += stat[3];
        metric_hops_exclude += stat[4];
        metric_dist_exclude += stat[5];
    }

    // State-based wrappers for after_search (gas_clean)
    template <size_t adedge_ver = 0, size_t adedge_size = 0, size_t shortcut_ver = 0, size_t shortcut_size = 0, size_t triprune_ver = 0, typename filter_t>
    void gas_clean_with_transition(const float* query, size_t k, filter_t* filter, size_t ef, bool use_bf = false) {
        // Acquire from SEARCHED_COUNTED state
        StatusList<dist_t>* status_list = status_list_pool_.acquire(StatusListPool<dist_t>::State::SEARCHED_COUNTED);
        
        // Do consumption
        gas_clean<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>(status_list, query, k, filter, ef, use_bf);
        
        // Transition to SEARCHED_CONSUMED state
        status_list_pool_.release(status_list, StatusListPool<dist_t>::State::SEARCHED_CONSUMED);
    }

    template <size_t adedge_ver = 0, size_t adedge_size = 0, size_t shortcut_ver = 0, size_t shortcut_size = 0, size_t triprune_ver = 0, typename filter_t>
    void gas_clean(StatusList<dist_t>* status_list_, const float* query, size_t k, filter_t* filter, size_t ef, bool use_bf = false) {
        if ((use_bf || status_list_->result_array_.size() < ef) && adedge_ver == 4) { // Literal TODO
            stdheap_t<dist_id_t> bf_results = bruteforce_search(query, ef, *filter);
            while (!bf_results.empty()) {
                dist_id_t bf_pair = bf_results.top();
                nodeid_t bf_id = bf_pair.second;
                if (!status_list_->get_visitbit(bf_id)) {
                    status_list_->set_status(bf_id, STATUS_FAILED);
                    status_list_->set_from(bf_id, bf_id, 1);
                    status_list_->push_gas_results(bf_id);
                }
                bf_results.pop();
            }
        }

        if constexpr (adedge_ver || shortcut_ver || triprune_ver)
            shortcut_storage_.template consume<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver, filter_t>(*status_list_, storage_, distf_, *filter, ef);
    }

    void gas_status_clean(StatusList<dist_t>* status_list_) const {
        status_list_->clear_all();
    }

    // State-based wrapper for status_clean
    void gas_status_clean_with_transition() const {
        // Acquire from SEARCHED_CONSUMED state
        StatusList<dist_t>* status_list = status_list_pool_.acquire(StatusListPool<dist_t>::State::SEARCHED_CONSUMED);
        
        // Clean it
        gas_status_clean(status_list);
        
        // Transition back to CLEARED state
        status_list_pool_.release(status_list, StatusListPool<dist_t>::State::CLEARED);
    }

    void clean_with_transition() const {
        // For baseline search that doesn't use gas, just clean directly
        StatusList<dist_t>* status_list = status_list_pool_.acquire(StatusListPool<dist_t>::State::SEARCHED_CONSUMED);
        clean(status_list);
        status_list_pool_.release(status_list, StatusListPool<dist_t>::State::CLEARED);
    }

    std::vector<size_t> get_statistics() const {
        std::vector<size_t> cgraph_stat = shortcut_storage_.get_statistics();
        size_t hops = metric_hops;
        size_t distance_computations = metric_distance_computations;
        size_t gas_prune_anchor_hit_ = metric_gas_prune_anchor_hit_;
        size_t gas_prune_anchor_miss_ = metric_gas_prune_anchor_miss_;
        size_t gas_prune_case1 = metric_gas_prune_case1;
        size_t gas_prune_case2 = metric_gas_prune_case2;
        size_t gas_prune_case3 = metric_gas_prune_case3;
        size_t gas_prune_case4 = metric_gas_prune_case4;
        size_t gas_prune_case5 = metric_gas_prune_case5;
        size_t hops_upper = metric_hops_upper_layer;
        size_t dist_upper = metric_dist_upper_layer;
        size_t hops_approach = metric_hops_approach;
        size_t dist_approach = metric_dist_approach;
        size_t hops_gather = metric_hops_gather;
        size_t dist_gather = metric_dist_gather;
        size_t hops_exclude = metric_hops_exclude;
        size_t dist_exclude = metric_dist_exclude;
        size_t pq_sort = metric_pq_sort;
        
        std::vector<size_t> ret = {
            hops, distance_computations,
            gas_prune_anchor_hit_, gas_prune_anchor_miss_,
            gas_prune_case1, gas_prune_case2, gas_prune_case3, gas_prune_case4, gas_prune_case5,
            hops_upper, dist_upper,
            hops_approach, dist_approach,
            hops_gather, dist_gather,
            hops_exclude, dist_exclude, pq_sort
        };

        ret.insert(ret.end(), cgraph_stat.begin(), cgraph_stat.end());
        return ret;
    }

    std::vector<size_t> renew() {
        std::vector<size_t> cgraph_stat = shortcut_storage_.renew();
        size_t hops = metric_hops;
        size_t distance_computations = metric_distance_computations;
        size_t gas_prune_anchor_hit_ = metric_gas_prune_anchor_hit_;
        size_t gas_prune_anchor_miss_ = metric_gas_prune_anchor_miss_;
        size_t gas_prune_case1 = metric_gas_prune_case1;
        size_t gas_prune_case2 = metric_gas_prune_case2;
        size_t gas_prune_case3 = metric_gas_prune_case3;
        size_t gas_prune_case4 = metric_gas_prune_case4;
        size_t gas_prune_case5 = metric_gas_prune_case5;
        size_t hops_upper = metric_hops_upper_layer;
        size_t dist_upper = metric_dist_upper_layer;
        size_t hops_approach = metric_hops_approach;
        size_t dist_approach = metric_dist_approach;
        size_t hops_gather = metric_hops_gather;
        size_t dist_gather = metric_dist_gather;
        size_t hops_exclude = metric_hops_exclude;
        size_t dist_exclude = metric_dist_exclude;
        size_t pq_sort = metric_pq_sort;
        metric_hops = 0;
        metric_distance_computations = 0;
        metric_gas_prune_anchor_hit_ = 0;
        metric_gas_prune_anchor_miss_ = 0;
        metric_gas_prune_case1 = 0;
        metric_gas_prune_case2 = 0;
        metric_gas_prune_case3 = 0;
        metric_gas_prune_case4 = 0;
        metric_gas_prune_case5 = 0;
        metric_hops_upper_layer = 0;
        metric_dist_upper_layer = 0;
        metric_hops_approach = 0;
        metric_dist_approach = 0;
        metric_hops_gather = 0;
        metric_dist_gather = 0;
        metric_hops_exclude = 0;
        metric_dist_exclude = 0;
        metric_pq_sort = 0;
        if (hops_approach != 0 && hops != hops_upper + hops_approach + hops_gather + hops_exclude) {
            throw std::runtime_error("Hops mismatch: " + std::to_string(hops) + " vs " +
                                     std::to_string(hops_upper) + " + " +
                                     std::to_string(hops_approach) + " + " +
                                     std::to_string(hops_gather) + " + " +
                                     std::to_string(hops_exclude));
        }
        std::vector<size_t> ret = {
            hops, distance_computations,
            gas_prune_anchor_hit_, gas_prune_anchor_miss_,
            gas_prune_case1, gas_prune_case2, gas_prune_case3, gas_prune_case4, gas_prune_case5,
            hops_upper, dist_upper,
            hops_approach, dist_approach,
            hops_gather, dist_gather,
            hops_exclude, dist_exclude, pq_sort
        };

        ret.insert(ret.end(), cgraph_stat.begin(), cgraph_stat.end());
        return ret;
    }

    bool save(const std::string path) {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file for saving index");
        }
        // Basic information
        const size_t magic_number = 0xCACADEDE;
        ofs.write((const char*) &magic_number, sizeof(magic_number)); // magic number
        ofs.write((const char*) &max_elements_, sizeof(max_elements_));
        ofs.write((const char*) &data_size_, sizeof(data_size_));
        ofs.write((const char*) &M_, sizeof(M_));
        ofs.write((const char*) &ef_construction_, sizeof(ef_construction_));

        ofs.write((const char*) &maxlevel_, sizeof(maxlevel_));
        ofs.write((const char*) &enterpoint_node_, sizeof(enterpoint_node_));
    
        // Graph Storage
        return storage_.save(ofs);
    }

    bool load(const std::string path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            return false;
        }
        size_t magic_number;
        ifs.read((char*) &magic_number, sizeof(size_t));
        if (magic_number != 0xCACADEDE) {
            throw std::runtime_error("Invalid magic number in the index file");
        }
        // Load basic information
        size_t max_elements;
        ifs.read((char*) &max_elements, sizeof(max_elements_));
        if (max_elements_ != max_elements) {
            throw std::runtime_error("Max elements mismatch in the index file");
        }
        size_t data_size;
        ifs.read((char*) &data_size, sizeof(data_size_));
        if (data_size_ != data_size) {
            throw std::runtime_error("Data size mismatch in the index file");
        }
        size_t M;
        ifs.read((char*) &M, sizeof(M_));
        if (M_ != M) {
            throw std::runtime_error("M mismatch in the index file");
        }
        ifs.read((char*) &ef_construction_, sizeof(ef_construction_));

        ifs.read((char*) &maxlevel_, sizeof(maxlevel_));
        ifs.read((char*) &enterpoint_node_, sizeof(enterpoint_node_));

        // Graph Storage
        return storage_.load(ifs);
    }

    void replace_meta(nodeid_t i, label_t new_label) {
        labeltype old_label = storage_.get_label_ptr(i)[0];

        size_t low32_mask = 0xFFFFFFFF;
        if ((new_label & low32_mask) != (old_label & low32_mask)) {
            throw std::runtime_error("Label mismatch for index " + std::to_string(i) + ": " +
                std::to_string(new_label) + " vs " + std::to_string(old_label));
        }

        label_lookup_[new_label] = label_lookup_[old_label];
        label_lookup_.erase(old_label);
        *storage_.get_label_ptr(i) = new_label;
    }


    void clear_shortcuts() {
        shortcut_storage_.clear_shortcuts();
    }
};
} // namespace gaslib
