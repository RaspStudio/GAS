#pragma once
#include "gas_strategy_impl.h"
#include <algorithm>
#include <vector>

namespace gaslib {


// Build tree and update containers
template<typename dist_t>
template <size_t adedge_ver, size_t adedge_size, size_t shortcut_ver, size_t shortcut_size, size_t triprune_ver, typename filter_t>
requires GASFilterConcept<filter_t>
void GASStorage<dist_t>::consume_and_clean(StatusList<dist_t>& status_list,
                        const GraphStorage& graph,
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
    GASTree<dist_t, filter_t> tree = GASTreeBuilder<adedge_ver, adedge_size, shortcut_ver, shortcut_size, triprune_ver>::build(status_list, graph, *this, distf, filter);

    // Track number of new edges added this round
    size_t new_additional_edges = 0;
    size_t new_shortcuts = 0;

    // 1) global permanent edges
    for (auto &e : tree.additional_edges_) {
        // add edges if they do not exist
        if (!graph.get_neighbors_l0(e.first)->contains(e.second)) {
            additional_edges_.add(e.first, e.second);
            new_additional_edges++;
        }
        if (!graph.get_neighbors_l0(e.second)->contains(e.first)) {
            additional_edges_.add(e.second, e.first);
            new_additional_edges++;
        }
    }

    // 3) per-tree overlay edges
    // if (tree_id < MAX_TREES) {
    for (auto &e : tree.shortcuts_) {
        // add edges if they do not exist
        if (!graph.get_neighbors_l0(e.first)->contains(e.second)) {
            shortcuts_.add(e.first, e.second);
            new_shortcuts++;
        }
        if (!graph.get_neighbors_l0(e.second)->contains(e.first)) {
            shortcuts_.add(e.second, e.first);
            new_shortcuts++;
        }
    }
    // }
    
    // // 4) per-tree flags
    // for (auto &f : tree.getFlags()) {
    //     overlayFlags_.add(f.node, f.value, tree_id);
    // }

    // Edge control mechanism - check and reduce edges if necessary
    size_t current_additional_edges = additional_edges_.count_all();
    size_t current_shortcuts = shortcuts_.count_all();

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

    // cleanup status
}

// Edge reduction function for additional_edges
template<typename dist_t>
template<typename filter_t>
void GASStorage<dist_t>::reduce_additional_edges(size_t target_reduction, const GraphStorage& graph, const GraphCompute<dist_t>& distf, const filter_t& filter) {
    if (target_reduction == 0) {
        return; // No reduction needed
    }
    
    // Find nodes with the most shortcuts (using shortcuts as proxy for activity)
    std::vector<std::pair<size_t, nodeid_t>> node_shortcut_counts;
    
    for (nodeid_t node = 0; node < max_elements_; ++node) {
        size_t shortcut_count = shortcuts_.size(node);
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
        if (additional_edges_.size(node) == 0) {
            continue;
        }
        
        // Find the neighbor with the largest vector distance
        const nodeid_t* neighbors = additional_edges_.data(node);
        size_t num_neighbors = additional_edges_.size(node);
        
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
        additional_edges_.remove_force(node, farthest_neighbor);
        
        edges_removed++;
    }
}

// Helper function for calculating neighbor similarity
template<typename dist_t>
template<typename filter_t>
int GASStorage<dist_t>::attribute_distance(nodeid_t node, nodeid_t neighbor, const GraphStorage& graph, const filter_t& filter) const {
    if constexpr (std::same_as<filter_t, RangeGASFilterFunctor>) {
        auto node_value = (*graph.get_label_ptr(node)) >> 32;
        auto neighbor_value = (*graph.get_label_ptr(neighbor)) >> 32;
        return node_value > neighbor_value ? node_value - neighbor_value : neighbor_value - node_value;
    } else if constexpr (std::same_as<filter_t, TagGASFilterFunctor>) {
        return (*graph.get_label_ptr(node)) == (*graph.get_label_ptr(neighbor)) ? 0 : 1;
    } else {
        throw std::runtime_error("Unsupported filter type for attribute_distance.");
    }
}

// Edge reduction function for shortcuts
template<typename dist_t>
template<typename filter_t>
void GASStorage<dist_t>::reduce_shortcuts(size_t target_reduction, const GraphStorage& graph, const GraphCompute<dist_t>& distf, const filter_t& filter) {
    if (target_reduction == 0) {
        return; // No reduction needed
    }
    
    // Find nodes with the most shortcuts
    std::vector<std::pair<size_t, nodeid_t>> node_shortcut_counts;
    
    for (nodeid_t node = 0; node < max_elements_; ++node) {
        size_t shortcut_count = shortcuts_.size(node);
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
        const auto& entries = shortcuts_.span(node);
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
            shortcuts_.remove_forced(node, neighbor);
            edges_removed++;
            remove_cnt--;
            if (remove_cnt == 0) {
                break;
            }
        }
    }
}

} // namespace gaslib