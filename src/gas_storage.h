#pragma once

#include "filter.h"
#include "status_list.h"

#include "gas_container.h"
#include "gas_strategy.h"

namespace gaslib {

template <typename dist_t>
class ShortcutStorage {
public:
    explicit ShortcutStorage(size_t max_elements, size_t max_additional_edges_per_node=8, size_t max_shortcuts_per_node=32)
        : max_elements_(max_elements),
          additional_edges_(max_elements),
          shortcuts_(max_elements),
          additional_edges_limit_(max_elements * max_additional_edges_per_node),
          shortcuts_limit_(max_elements * max_shortcuts_per_node),
          max_sc_per_node_(max_shortcuts_per_node)
    {}

    virtual ~ShortcutStorage() = default;

    // Get statistics without resetting containers
    std::vector<size_t> get_statistics() const {
        std::vector<size_t> ret;
        ret.push_back(additional_edges().count_all());
        ret.push_back(shortcuts().count_all());
        return ret;
    }

    // Reset all containers
    virtual std::vector<size_t> renew() {
        std::vector<size_t> ret;
        ret.push_back(additional_edges().count_all());
        additional_edges_ = ShortcutList<nodeid_t>(max_elements_);
        ret.push_back(shortcuts().count_all());
        shortcuts_      = RefShortcutList<nodeid_t>(max_elements_);
        return ret;
    }

    // Build tree and update containers
    template <size_t adedge_ver = 0, size_t adedge_size = 0, size_t shortcut_ver = 0, size_t shortcut_size = 0, size_t triprune_ver = 0, typename filter_t, GraphConcept graph_t = GraphStorage>
    requires GasFilterConcept<filter_t>
    void consume(StatusList<dist_t>& status_list,
                         const graph_t& graph,
                         const GraphCompute<dist_t>& distf,
                         const filter_t& filter, size_t ef);

    // Read-only accessors
    const ShortcutList<nodeid_t>&         additional_edges() const { return additional_edges_; }
    const RefShortcutList<nodeid_t>&      shortcuts() const { return shortcuts_; }

    // Clear shortcuts for widesheet dataset
    void clear_shortcuts();

private:
    size_t max_elements_;

    ShortcutList<nodeid_t>         additional_edges_;
    RefShortcutList<nodeid_t> shortcuts_;
    
    size_t additional_edges_limit_;
    size_t shortcuts_limit_;
    size_t max_sc_per_node_;

protected:
    // ---- Virtual mutation hooks ----
    // Override these in subclasses to add sync / logging / custom storage.
    // Called only during consume() (offline), never on the hot search path.
    virtual void add_additional_edge(nodeid_t u, nodeid_t v)    { additional_edges_.add(u, v); }
    virtual void remove_additional_edge(nodeid_t u, nodeid_t v) { additional_edges_.remove_force(u, v); }
    virtual void add_shortcut(nodeid_t u, nodeid_t v)           { shortcuts_.add(u, v); }
    virtual void remove_shortcut(nodeid_t u, nodeid_t v)        { shortcuts_.remove_forced(u, v); }

    virtual size_t ae_cache_capacity(nodeid_t) const { return SIZE_MAX; }

private:
    // Edge reduction functions
    template<typename filter_t, GraphConcept graph_t = GraphStorage>
    void reduce_additional_edges(size_t target_reduction, const graph_t& graph, const GraphCompute<dist_t>& distf, const filter_t& filter);
    
    template<typename filter_t, GraphConcept graph_t = GraphStorage>
    void reduce_shortcuts(size_t target_reduction, const graph_t& graph, const GraphCompute<dist_t>& distf, const filter_t& filter);
    
    // Helper function for calculating neighbor similarity
    template<typename filter_t, GraphConcept graph_t = GraphStorage>
    int attribute_distance(nodeid_t node, nodeid_t neighbor, const graph_t& graph, const filter_t& filter) const;

};

// ==========================================================================
// ==========================================================================
template <typename dist_t>
class GASShortcutStorage : public ShortcutStorage<dist_t> {
    GASStorage* gas_{ nullptr };

public:
    GASShortcutStorage(size_t max_elements, GASStorage* gas,
                    size_t max_additional_edges_per_node = 8,
                    size_t max_shortcuts_per_node = 32)
        : ShortcutStorage<dist_t>(max_elements, max_additional_edges_per_node, max_shortcuts_per_node)
        , gas_(gas) {}

    void bind_gas(GASStorage* gas) { gas_ = gas; }

    std::vector<size_t> renew() override {
        auto ret = ShortcutStorage<dist_t>::renew();
        if (gas_) {
            for (size_t i = 0; i < gas_->max_elements_; ++i) {
                auto* nl = gas_->get_neighbors_l0(i);
                nodeid_t n_orig = gas_->get_original_neighbor_count(i);
                nl->len_ = n_orig;
                gas_->get_sc_l0(i)->len_ = 0;
            }
        }
        return ret;
    }

protected:
    void add_additional_edge(nodeid_t u, nodeid_t v) override {
        ShortcutStorage<dist_t>::add_additional_edge(u, v);
        if (gas_) gas_->append_ae_l0(u, v);
    }
    void remove_additional_edge(nodeid_t u, nodeid_t v) override {
        ShortcutStorage<dist_t>::remove_additional_edge(u, v);
        if (gas_) gas_->remove_ae_l0(u, v);
    }
    void add_shortcut(nodeid_t u, nodeid_t v) override {
        ShortcutStorage<dist_t>::add_shortcut(u, v);
        if (gas_) gas_->append_sc_l0(u, v);
    }
    void remove_shortcut(nodeid_t u, nodeid_t v) override {
        ShortcutStorage<dist_t>::remove_shortcut(u, v);
        if (gas_) gas_->remove_sc_l0(u, v);
    }

    size_t ae_cache_capacity(nodeid_t u) const override {
        return gas_ ? gas_->get_merged_capacity() - gas_->get_original_neighbor_count(u) : SIZE_MAX;
    }
};

template <typename dist_t, typename filter_t>
requires GasFilterConcept<filter_t>
struct GasTree {
    std::vector<id_pair_t> additional_edges_;
    std::vector<id_pair_t> shortcuts_;

    explicit GasTree() :
        additional_edges_(),
        shortcuts_() {}

    explicit GasTree(std::vector<id_pair_t> additional_edges,
                std::vector<id_pair_t> shortcuts) :
        additional_edges_(std::move(additional_edges)),
        shortcuts_(std::move(shortcuts)) {}

    GasTree(const GasTree&) = delete;
    GasTree(GasTree&& other) noexcept :
        additional_edges_(std::move(other.additional_edges_)),
        shortcuts_(std::move(other.shortcuts_)) {}

    GasTree& operator=(const GasTree&) = delete;
    GasTree& operator=(GasTree&& other) noexcept {
        if (this != &other) {
            additional_edges_ = std::move(other.additional_edges_);
            shortcuts_ = std::move(other.shortcuts_);
        }
        return *this;
    }
};

template <typename dist_t, GraphConcept graph_t>
bool select_neighbor(nodeid_t candidate,
                        const graph_t& graph, 
                        const GraphCompute<dist_t>& distf,
                        const nodeid_t id) {
    
    auto original = graph.get_l0_span(id);
    dist_t cand_to_id = distf.get_distance(graph.get_data_ptr(candidate), graph.get_data_ptr(id));
    for (const nodeid_t& existing : original) {
        dist_t exist_to_id = distf.get_distance(graph.get_data_ptr(existing), graph.get_data_ptr(id));
        if (exist_to_id > cand_to_id) {
            continue;
        }
        dist_t exist_to_cand = distf.get_distance(graph.get_data_ptr(existing), graph.get_data_ptr(candidate));
        if (exist_to_cand < cand_to_id && exist_to_id < cand_to_id) {
            return false;
        }
    }
    // todo check cgraph
    return true;
}

template <typename dist_t, GraphConcept graph_t>
auto select_neighbors(std::vector<std::pair<dist_t, nodeid_t>> candidates, 
                        const graph_t& graph, 
                        const GraphCompute<dist_t>& distf,
                        const nodeid_t id, const size_t M, std::vector<id_pair_t>* pruned = nullptr) -> std::vector<id_pair_t> {
    auto original = graph.get_l0_span(id);
    std::vector<std::pair<dist_t, tableint>> return_list;
    std::vector<id_pair_t> ret_edges;

    for (const auto& candidate : candidates) {
        tableint candidate_id = candidate.second;
        dist_t cand_to_id = candidate.first;
        bool good = true;

        if (std::find(original.begin(), original.end(), candidate_id) != original.end()) {
            continue;
        }

        // todo check cgraph

        for (const nodeid_t& existing : original) {
            dist_t exist_to_id = distf.get_distance(graph.get_data_ptr(existing), graph.get_data_ptr(id));
            if (exist_to_id > cand_to_id) {
                continue;
            }
            dist_t exist_to_cand = distf.get_distance(graph.get_data_ptr(existing), graph.get_data_ptr(candidate_id));
            if (exist_to_cand < cand_to_id && exist_to_id < cand_to_id) {
                good = false;
                if (pruned) pruned->emplace_back(existing, candidate_id);
            }
        }

        for (const auto& existing_new : ret_edges) {
            dist_t existnew_to_cand = distf.get_distance(graph.get_data_ptr(existing_new.second), graph.get_data_ptr(candidate_id));
            if (existnew_to_cand < cand_to_id && existing_new.first < cand_to_id) {
                good = false;
                if (pruned) pruned->emplace_back(existing_new.second, candidate_id);
            }
        }

        if (good) {
            ret_edges.emplace_back(id, candidate_id);
            return_list.emplace_back(cand_to_id, candidate_id);
            if (ret_edges.size() >= M) {
                break;
            }
        }
    }

    return ret_edges;
}


template <typename dist_t, typename filter_t, GraphConcept graph_t>
auto special_additional_edges(const std::vector<nodeid_t>& special_nodes,
                            const StatusList<dist_t>& status_list,
                            const graph_t& graph,
                            const GraphCompute<dist_t>& distf,
                            const filter_t& filter, const nodeid_t ep, size_t ae_size = 1) -> std::vector<id_pair_t> {
    std::vector<id_pair_t> out;   
    for (nodeid_t not_reachable : special_nodes) {
        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, ComparePairFirst<dist_t, nodeid_t>> pq;
        pq.emplace(-distf.get_distance(graph.get_data_ptr(not_reachable), graph.get_data_ptr(ep)), ep);
        nodeid_t cur;
        do {
            cur = pq.top().second;
            for (nodeid_t nbr : graph.get_l0_span(cur)) {
                dist_t dist_to_nreach = distf.get_distance(graph.get_data_ptr(nbr), graph.get_data_ptr(not_reachable));
                pq.emplace(-dist_to_nreach, nbr);
            }
        } while (pq.top().second != cur);
        std::vector<std::pair<dist_t, nodeid_t>> candidates;
        while (!pq.empty()) {
            auto top = pq.top();
            pq.pop();
            candidates.emplace_back(-top.first, top.second);
        }
        std::vector<id_pair_t> nbrs = select_neighbors(candidates, graph, distf, not_reachable, ae_size);
        for (const auto& id_pair : nbrs) {
            id_pair_t key = (id_pair.first > id_pair.second) ? std::make_pair(id_pair.second, id_pair.first) : std::make_pair(id_pair.first, id_pair.second);
            if (std::find(out.begin(), out.end(), key) == out.end()) {
                out.emplace_back(key);
            }
        }
    }
    return out;
}

template <typename dist_t, typename filter_t, GraphConcept graph_t>
void add_monotonic_edges(const StatusList<dist_t>& status_list,
                            const graph_t& graph,
                            const GraphCompute<dist_t>& distf,
                            const filter_t& filter, size_t ae_size, nodeid_t res, nodeid_t& ep, std::vector<id_pair_t>& out) {
    bool monotonic = true;
    std::vector<std::pair<dist_t, nodeid_t>> cand_in_path;
    {
        nodeid_t steps = 0;
        nodeid_t cur = res;
        std::vector<std::pair<dist_t, nodeid_t>> path;
        std::set<nodeid_t> visited;
        while (status_list.get_from(cur) != cur) {
            nodeid_t next = status_list.get_from(cur);
            path.emplace_back(status_list.get_dist(next), next);
            if (visited.find(next) != visited.end()) {
                throw std::runtime_error("Cycle detected in path for node " + std::to_string(res));
            }
            visited.insert(next);
            cur = next;
            steps++;
        }

        if (ep == INVALID_NODE_ID) {
            ep = cur;
        } else if (ep != cur) {
            throw std::runtime_error("Root node mismatch, expected " + std::to_string(ep) + ", got " + std::to_string(cur));
        }

        dist_t thres = status_list.get_dist(res);
        auto rit = path.rbegin();

        while (rit != path.rend() && (*rit).first > thres) {
            rit++;
        }
        for (; rit != path.rend(); rit++) {
            dist_t dist_to_query = (*rit).first;
            if (dist_to_query > thres) {
                monotonic = false;
                break;
            }
            cand_in_path.emplace_back(distf.get_distance(
                graph.get_data_ptr(res), graph.get_data_ptr((*rit).second)
            ), (*rit).second);
        }
    }      

    if (monotonic || cand_in_path.empty()) {
        return;
    }

    std::sort(cand_in_path.begin(), cand_in_path.end(), [](const std::pair<dist_t, nodeid_t>& a, const std::pair<dist_t, nodeid_t>& b) {
        return a.first < b.first;
    });

    std::vector<id_pair_t> nbrs = select_neighbors(cand_in_path, graph, distf, res, ae_size);
    for (const auto& id_pair : nbrs) {
        id_pair_t key = (id_pair.first > id_pair.second) ? std::make_pair(id_pair.second, id_pair.first) : std::make_pair(id_pair.first, id_pair.second);
        if (std::find(out.begin(), out.end(), key) == out.end()) {
            out.emplace_back(key);
        }
    }
}

template <typename dist_t, typename filter_t, GraphConcept graph_t>
auto naive_additional_edges_v4(const StatusList<dist_t>& status_list,
                            const graph_t& graph,
                            const GraphCompute<dist_t>& distf,
                            const filter_t& filter, size_t ae_size = 1) -> std::vector<id_pair_t> {
    const auto& results = status_list.result_array_;
    const auto& valids = status_list.valid_array_;
    std::vector<id_pair_t> out;
    std::vector<nodeid_t> special_nodes;
    nodeid_t ep = status_list.l0_ep_;

    if (ep == INVALID_NODE_ID) throw std::runtime_error("Endpoint is not set, please set it before calling this function.");

    for (nodeid_t res : results) {
        if (!status_list.get_visitbit(res)) {
            special_nodes.push_back(res);
            continue;
        }
        add_monotonic_edges(status_list, graph, distf, filter, ae_size, res, ep, out);
    }

    for (nodeid_t res : valids) {
        if (!status_list.get_visitbit(res)) {
            continue;
        }
        add_monotonic_edges(status_list, graph, distf, filter, ae_size, res, ep, out);
    }

    std::vector<id_pair_t> special_edges = 
        special_additional_edges(special_nodes, status_list, graph, distf, filter, ep, ae_size);
    out.insert(out.end(), special_edges.begin(), special_edges.end());

    return out;
}


template <typename dist_t, typename filter_t, GraphConcept graph_t>
auto nn_shortcuts_v5(const StatusList<dist_t>& status_list,
                           const graph_t& graph,
                           const GraphCompute<dist_t>& distf,
                           const filter_t& filter, const size_t sc_size = 2) -> std::vector<id_pair_t> {
    const auto& results = status_list.result_array_;
    size_t n = results.size();
    
    std::unordered_set<nodeid_t> result_set(results.begin(), results.end());

    std::vector<id_pair_t> edges;
    edges.reserve(n * 2);

    for (nodeid_t curr_id : results) {
        while (true) {
            nodeid_t parent = status_list.get_from(curr_id);
            if (parent == curr_id) break;
            if (result_set.count(parent)) {
                id_pair_t ep = (parent < curr_id)
                                 ? std::make_pair(parent, curr_id)
                                 : std::make_pair(curr_id, parent);
                edges.emplace_back(ep);
                break;
            }
            curr_id = parent;
        }
    }

    std::unordered_map<id_pair_t, dist_t> distances;
    distances.reserve(n * (n - 1) / 2);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            nodeid_t id1 = results[i];
            nodeid_t id2 = results[j];
            id_pair_t key = (id1 < id2)
                             ? std::make_pair(id1, id2)
                             : std::make_pair(id2, id1);
            dist_t d = distf.get_distance(
                graph.get_data_ptr(id1), graph.get_data_ptr(id2)
            );
            distances.emplace(key, d);
        }
    }

    for (nodeid_t id : result_set) {
        std::vector<std::pair<dist_t, nodeid_t>> candidates;
        for (nodeid_t oid : result_set) {
            if (oid == id) continue;
            dist_t d = distances.at(
                (id < oid) ? std::make_pair(id, oid) : std::make_pair(oid, id)
            );
            candidates.emplace_back(d, oid);
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (size_t i = 0; i < candidates.size() && i < sc_size; ++i) {
            nodeid_t nbr = candidates[i].second;
            id_pair_t key = (id < nbr) ? std::make_pair(id, nbr) : std::make_pair(nbr, id);
            if (std::find(edges.begin(), edges.end(), key) == edges.end()) {
                edges.emplace_back(key);
            }
        }
    }

    return edges;
}


template <size_t adedge_ver, size_t adedge_size, size_t shortcut_ver, size_t shortcut_size, size_t triprune_ver>
template<typename dist_t, typename filter_t, GraphConcept graph_t>
auto GasTreeBuilder<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>::build(
        const StatusList<dist_t>& status_list,
        const graph_t& graph,
        const ShortcutStorage<dist_t>& cgraph,
        const GraphCompute<dist_t>& distf,
        const filter_t& filter) -> GasTree<dist_t, filter_t> {
    std::vector<id_pair_t> additional_edges;
    if constexpr (adedge_ver == 1) {
        additional_edges = naive_additional_edges_v1(status_list, graph, distf, filter, adedge_size);
    } else if constexpr (adedge_ver == 2) {
        additional_edges = naive_additional_edges_v2(status_list, graph, distf, filter, adedge_size);
    } else if constexpr (adedge_ver == 3) {
        additional_edges = naive_additional_edges_v3(status_list, graph, distf, filter, adedge_size);
    } else if constexpr (adedge_ver == 4) {
        additional_edges = naive_additional_edges_v4(status_list, graph, distf, filter, adedge_size);
    }

    std::vector<id_pair_t> shortcuts;
    if constexpr (shortcut_ver == 1) {
        shortcuts = naive_shortcuts_v1(status_list, graph, distf, filter);
    } else if constexpr (shortcut_ver == 2) {
        shortcuts = pruned_shortcuts_v2(status_list, graph, distf, filter);
    } else if constexpr (shortcut_ver == 3) {
        shortcuts = nnbased_shortcuts_v3(status_list, graph, distf, filter);
    } else if constexpr (shortcut_ver == 4) {
        shortcuts = selnbr_shortcuts_v4(status_list, graph, distf, filter, shortcut_size);
    } else if constexpr (shortcut_ver == 5) {
        shortcuts = nn_shortcuts_v5(status_list, graph, distf, filter, shortcut_size);
    }

    return GasTree<dist_t, filter_t>(
        std::move(additional_edges),
        std::move(shortcuts)
    );
}

// Build tree and update containers
template<typename dist_t>
template <size_t adedge_ver, size_t adedge_size, size_t shortcut_ver, size_t shortcut_size, size_t triprune_ver, typename filter_t, GraphConcept graph_t>
requires GasFilterConcept<filter_t>
void ShortcutStorage<dist_t>::consume(StatusList<dist_t>& status_list,
                        const graph_t& graph,
                        const GraphCompute<dist_t>& distf,
                        const filter_t& filter, size_t ef) {
    // allocate tree id
    // size_t tree_id = allocateTree();

    // sort result before constructing tree
    if (status_list.result_array_.size() < ef) {
        // throw std::runtime_error("Result array size is less than ef, cannot build tree.");
        // TODO: fix graph
        std::cerr << "Warning: Result array size is less than ef, cannot build tree." << std::endl;
        return;
    }

    status_list.result_array_full_hop_ = status_list.get_hop(status_list.result_array_[ef]);
    std::sort(status_list.result_array_.begin(), status_list.result_array_.end(),
                [&status_list](nodeid_t a, nodeid_t b) {
                    return status_list.get_dist(a) < status_list.get_dist(b);
                });
    
    // if result_array exceed max_size, move the last elements to the valid array
    if (status_list.result_array_.size() > ef) {
        size_t valid_size = ef;
        status_list.valid_array_.assign(status_list.result_array_.begin() + valid_size, status_list.result_array_.end());
        status_list.result_array_.resize(valid_size);
    }

    // Build tree using modular builder
    GasTree<dist_t, filter_t> tree = GasTreeBuilder<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>::build(status_list, graph, *this, distf, filter);

    // Track number of new edges added this round
    size_t new_additional_edges = 0;
    size_t new_shortcuts = 0;

    // 1) global permanent edges
    for (auto &e : tree.additional_edges_) {
        auto try_add_ae = [&](nodeid_t u, nodeid_t v) {
            if (std::ranges::find(graph.get_l0_span(u), v) != graph.get_l0_span(u).end())
                return;
            if (ae_cache_capacity(u) != SIZE_MAX && additional_edges().size(u) >= ae_cache_capacity(u)) {
                const nodeid_t* ae = additional_edges().data(u);
                size_t n = additional_edges().size(u);
                nodeid_t farthest = ae[0];
                dist_t max_d = distf.get_distance(graph.get_data_ptr(u), graph.get_data_ptr(farthest));
                for (size_t j = 1; j < n; ++j) {
                    dist_t d = distf.get_distance(graph.get_data_ptr(u), graph.get_data_ptr(ae[j]));
                    if (d > max_d) { max_d = d; farthest = ae[j]; }
                }
                remove_additional_edge(u, farthest);
            }
            add_additional_edge(u, v);
            new_additional_edges++;
        };
        try_add_ae(e.first, e.second);
        try_add_ae(e.second, e.first);
    }

    // 3) per-tree overlay edges
    // if (tree_id < MAX_TREES) {
    for (auto &e : tree.shortcuts_) {
        auto try_add_sc = [&](nodeid_t u, nodeid_t v) {
            if (std::ranges::find(graph.get_l0_span(u), v) != graph.get_l0_span(u).end())
                return;
            if (shortcuts().size(u) >= max_sc_per_node_) {
                const auto& entries = shortcuts().span(u);
                nodeid_t worst = entries[0].value;
                dist_t worst_dist = distf.get_distance(graph.get_data_ptr(u), graph.get_data_ptr(worst));
                int worst_atti = attribute_distance(u, worst, graph, filter);
                for (const auto& entry : entries) {
                    dist_t d = distf.get_distance(graph.get_data_ptr(u), graph.get_data_ptr(entry.value));
                    int atti = attribute_distance(u, entry.value, graph, filter);
                    if (atti > worst_atti || (atti == worst_atti && d > worst_dist)) {
                        worst = entry.value;
                        worst_dist = d;
                        worst_atti = atti;
                    }
                }
                remove_shortcut(u, worst);
            }
            add_shortcut(u, v);
            new_shortcuts++;
        };
        try_add_sc(e.first, e.second);
        try_add_sc(e.second, e.first);
    }
    // }
    
    // // 4) per-tree flags
    // for (auto &f : tree.getFlags()) {
    //     overlayFlags_.add(f.node, f.value, tree_id);
    // }

    // Edge control mechanism - check and reduce edges if necessary
    size_t current_additional_edges = additional_edges().count_all();
    size_t current_shortcuts = shortcuts().count_all();

    // Check and reduce additional_edges if necessary
    if (current_additional_edges > additional_edges_limit_) {
        size_t target_reduction = current_additional_edges - additional_edges_limit_;
        reduce_additional_edges(target_reduction, graph, distf, filter);
    }

    // Check and reduce shortcuts if necessary
    if (current_shortcuts > shortcuts_limit_) {
        size_t target_reduction = current_shortcuts - shortcuts_limit_;
        reduce_shortcuts(target_reduction, graph, distf, filter);
    }
}

// Edge reduction function for additional_edges
template<typename dist_t>
template<typename filter_t, GraphConcept graph_t>
void ShortcutStorage<dist_t>::reduce_additional_edges(size_t target_reduction, const graph_t& graph, const GraphCompute<dist_t>& distf, const filter_t& filter) {
    if (target_reduction == 0) {
        return; // No reduction needed
    }
    
    // Find nodes with the most shortcuts (using shortcuts as proxy for activity)
    std::vector<std::pair<size_t, nodeid_t>> node_shortcut_counts;
    
    for (nodeid_t node = 0; node < max_elements_; ++node) {
        size_t shortcut_count = shortcuts().size(node);
        if (shortcut_count > 0) {
            node_shortcut_counts.emplace_back(shortcut_count, node);
        }
    }
    
    if (node_shortcut_counts.empty()) {
        return; // No nodes with shortcuts to guide the reduction
    }
    
    // Sort by shortcut count in descending order to find the most active nodes
    std::partial_sort(
        node_shortcut_counts.begin(), 
        node_shortcut_counts.begin() + std::min(target_reduction, node_shortcut_counts.size()), 
        node_shortcut_counts.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Select nodes to process based on those with most shortcuts
    size_t nodes_to_process = std::min(target_reduction, node_shortcut_counts.size());
    
    size_t edges_removed = 0;
    for (size_t i = 0; i < nodes_to_process && edges_removed < target_reduction; ++i) {
        nodeid_t node = node_shortcut_counts[i].second;
        
        // Skip if this node has no additional edges
        if (additional_edges().size(node) == 0) {
            continue;
        }
        
        // Find the neighbor with the largest vector distance
        const nodeid_t* neighbors = additional_edges().data(node);
        size_t num_neighbors = additional_edges().size(node);
        
        if (num_neighbors == 0) {
            continue;
        }
        
        nodeid_t farthest_neighbor = neighbors[0];
        dist_t max_distance = distf.get_distance(graph.get_data_ptr(node), graph.get_data_ptr(neighbors[0]));
        
        // Find the neighbor with maximum distance
        for (size_t j = 1; j < num_neighbors; ++j) {
            nodeid_t neighbor = neighbors[j];
            dist_t distance = distf.get_distance(graph.get_data_ptr(node), graph.get_data_ptr(neighbor));
            if (distance > max_distance) {
                max_distance = distance;
                farthest_neighbor = neighbor;
            }
        }
        
        // Remove the farthest neighbor using the new remove_force method
        remove_additional_edge(node, farthest_neighbor);
        
        edges_removed++;
    }
}

// Helper function for calculating neighbor similarity
template<typename dist_t>
template<typename filter_t, GraphConcept graph_t>
int ShortcutStorage<dist_t>::attribute_distance(nodeid_t node, nodeid_t neighbor, const graph_t& graph, const filter_t& filter) const {
    if constexpr (std::same_as<filter_t, RangeGasFilterFunctor>) {
        auto node_value = (*graph.get_label_ptr(node)) >> 32;
        auto neighbor_value = (*graph.get_label_ptr(neighbor)) >> 32;
        return node_value > neighbor_value ? node_value - neighbor_value : neighbor_value - node_value;
    } else if constexpr (std::same_as<filter_t, TagGasFilterFunctor>) {
        return (*graph.get_label_ptr(node)) == (*graph.get_label_ptr(neighbor)) ? 0 : 1;
    } else {
        throw std::runtime_error("Unsupported filter type for attribute_distance.");
    }
}

// Edge reduction function for shortcuts
template<typename dist_t>
template<typename filter_t, GraphConcept graph_t>
void ShortcutStorage<dist_t>::reduce_shortcuts(size_t target_reduction, const graph_t& graph, const GraphCompute<dist_t>& distf, const filter_t& filter) {
    if (target_reduction == 0) {
        return; // No reduction needed
    }
    
    // Find nodes with the most shortcuts
    std::vector<std::pair<size_t, nodeid_t>> node_shortcut_counts;
    
    for (nodeid_t node = 0; node < max_elements_; ++node) {
        size_t shortcut_count = shortcuts().size(node);
        if (shortcut_count > 0) {
            node_shortcut_counts.emplace_back(shortcut_count, node);
        }
    }
    
    if (node_shortcut_counts.empty()) {
        return; // No shortcuts to reduce
    }
    
    // Sort by shortcut count in descending order
    std::partial_sort(
        node_shortcut_counts.begin(), 
        node_shortcut_counts.begin() + std::min(target_reduction, node_shortcut_counts.size()), 
        node_shortcut_counts.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Select nodes to process based on those with most shortcuts
    // Process nodes proportionally to the reduction needed
    size_t nodes_to_process = std::min(target_reduction, node_shortcut_counts.size());
    
    size_t edges_removed = 0;
    for (size_t i = 0; i < nodes_to_process && edges_removed < target_reduction; ++i) {
        nodeid_t node = node_shortcut_counts[i].second;
        size_t remove_cnt = (target_reduction + nodes_to_process - 1) / nodes_to_process; // Ceiling division

        // Create a vector of neighbors with their similarity scores
        std::vector<std::tuple<int, dist_t, nodeid_t>> neighbor_scores; // (atti_dist, distance, neighbor)
       
        // Get all shortcuts for this node
        const auto& entries = shortcuts().span(node);
        if (entries.empty()) {
            continue; // Skip if no shortcuts (shouldn't happen, but safety check)
        }

        // Calculate scores for each neighbor
        for (const auto& entry : entries) {
            nodeid_t neighbor = entry.value;
            dist_t distance = distf.get_distance(graph.get_data_ptr(node), graph.get_data_ptr(neighbor));
            int atti_dist = attribute_distance(node, neighbor, graph, filter);
            neighbor_scores.emplace_back(atti_dist, distance, neighbor);
        }

        // Sort neighbors by attribute distance (ascending) and then by distance (ascending)
        std::sort(neighbor_scores.begin(), neighbor_scores.end(),
                  [](const auto& a, const auto& b) {
                      if (std::get<0>(a) != std::get<0>(b)) {
                          return std::get<0>(a) < std::get<0>(b); // Attribute distance ascending
                      }
                      return std::get<1>(a) < std::get<1>(b); // Distance ascending
                  });

        // Remove the least similar neighbors first
        for (const auto& score : neighbor_scores) {
            if (edges_removed >= target_reduction) {
                break;
            }
            nodeid_t neighbor = std::get<2>(score);
            remove_shortcut(node, neighbor);
            edges_removed++;
            remove_cnt--;
            if (remove_cnt == 0) {
                break;
            }
        }
    }
}


template<typename dist_t>
void ShortcutStorage<dist_t>::clear_shortcuts() {
    shortcuts_ = RefShortcutList<nodeid_t>(max_elements_);
}

} // namespace gaslib
