#pragma once
#include "filter.h"
#include "types.h"

#include "gas_storage.h"

namespace gaslib {

template <typename dist_t, typename filter_t>
requires GASFilterConcept<filter_t>
struct GASTree {
    
    std::vector<id_pair_t> additional_edges_;
    
    std::vector<id_pair_t> shortcuts_;

    explicit GASTree() :
        additional_edges_(),
        shortcuts_() {}

    explicit GASTree(std::vector<id_pair_t> additional_edges,
                std::vector<id_pair_t> shortcuts) :
        additional_edges_(std::move(additional_edges)),
        shortcuts_(std::move(shortcuts)) {}

    GASTree(const GASTree&) = delete; 
    GASTree(GASTree&& other) noexcept :
        additional_edges_(std::move(other.additional_edges_)),
        shortcuts_(std::move(other.shortcuts_)) {}

    GASTree& operator=(const GASTree&) = delete; 
    GASTree& operator=(GASTree&& other) noexcept {
        if (this != &other) {
            additional_edges_ = std::move(other.additional_edges_);
            shortcuts_ = std::move(other.shortcuts_);
        }
        return *this;
    }
};



template <typename dist_t>
auto select_neighbors(std::vector<std::pair<dist_t, nodeid_t>> candidates, 
                        const GraphStorage& graph, 
                        const GraphCompute<dist_t>& distf,
                        const nodeid_t id, const size_t M, std::vector<id_pair_t>* pruned = nullptr) -> std::vector<id_pair_t> {
    std::span<nodeid_t> original = graph.get_neighbors_l0(id)->span();
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




template <typename dist_t, typename filter_t>
auto special_additional_edges(const std::vector<nodeid_t>& special_nodes,
                            const StatusList<dist_t>& status_list,
                            const GraphStorage& graph,
                            const GraphCompute<dist_t>& distf,
                            const filter_t& filter, const nodeid_t ep, size_t ae_size = 1) -> std::vector<id_pair_t> {
    
    std::vector<id_pair_t> out;   
    
    for (nodeid_t not_reachable : special_nodes) {
        
        std::priority_queue<std::pair<dist_t, nodeid_t>, std::vector<std::pair<dist_t, nodeid_t>>, ComparePairFirst<dist_t, nodeid_t>> pq;
        
        pq.emplace(-distf.get_distance(graph.get_data_ptr(not_reachable), graph.get_data_ptr(ep)), ep);
        
        nodeid_t cur;
        do {
            cur = pq.top().second; 
            for (nodeid_t nbr : graph.get_neighbors_l0(cur)->span()) {
                
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

template <typename dist_t, typename filter_t>
void add_monotonic_edges(const StatusList<dist_t>& status_list,
                            const GraphStorage& graph,
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


template <typename dist_t, typename filter_t>
auto naive_additional_edges_v4(const StatusList<dist_t>& status_list,
                            const GraphStorage& graph,
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


template <typename dist_t, typename filter_t>
auto nn_shortcuts_v5(const StatusList<dist_t>& status_list,
                           const GraphStorage& graph,
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
template<typename dist_t, typename filter_t>
auto GASTreeBuilder<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>::build(
        const StatusList<dist_t>& status_list,
        const GraphStorage& graph,
        const GASStorage<dist_t>& cgraph,
        const GraphCompute<dist_t>& distf,
        const filter_t& filter) -> GASTree<dist_t, filter_t> {
    
    std::vector<id_pair_t> additional_edges;
    if constexpr (adedge_ver == 4) {
        
        additional_edges = naive_additional_edges_v4(status_list, graph, distf, filter, adedge_size);
    }

    
    std::vector<id_pair_t> shortcuts;
    if constexpr (shortcut_ver == 5) {
        
        shortcuts = nn_shortcuts_v5(status_list, graph, distf, filter, shortcut_size);
    }

    return GASTree<dist_t, filter_t>(
        std::move(additional_edges),
        std::move(shortcuts)
    );
}

} // namespace gaslib