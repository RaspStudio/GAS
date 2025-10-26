#pragma once
#include <map>

#include "filter.h"
#include "status_list.h"

#include "gas_container.h"
#include "gas_strategy.h"

namespace gaslib {

template <typename dist_t>
class GASStorage {
public:
    explicit GASStorage(size_t max_elements)
        : max_elements_(max_elements),
          additional_edges_(max_elements),
          shortcuts_(max_elements),
          additional_edges_limit_(max_elements * MAX_ADDITIONAL_EDGES_PER_NODE),
          shortcuts_limit_(max_elements * MAX_SHORTCUTS_PER_NODE)
        //   overlayFlags_(max_elements)
    {}

    // Get statistics without resetting containers
    std::vector<size_t> get_statistics() const {
        std::vector<size_t> ret;
        ret.push_back(additional_edges_.count_all());
        ret.push_back(shortcuts_.count_all());
        return ret;
    }

    // Reset all containers
    std::vector<size_t> renew() {
        std::vector<size_t> ret;
        ret.push_back(additional_edges_.count_all());
        additional_edges_ = Cracks<nodeid_t>(max_elements_);
        ret.push_back(shortcuts_.count_all());
        shortcuts_      = RefCracks<nodeid_t>(max_elements_);
        // overlayFlags_   = OptionalCracks<TFlag>(max_elements_);
        treeSlots_.reset();
        return ret;
    }

    // Build tree and update containers
    template <size_t adedge_ver = 0, size_t adedge_size = 0, size_t shortcut_ver = 0, size_t shortcut_size = 0, size_t triprune_ver = 0, typename filter_t>
    requires GASFilterConcept<filter_t>
    void consume_and_clean(StatusList<dist_t>& status_list,
                         const GraphStorage& graph,
                         const GraphCompute<dist_t>& distf,
                         const filter_t& filter, size_t ef);

    // Read-only accessors
    const Cracks<nodeid_t>&         additional_edges() const { return additional_edges_; }
    const RefCracks<nodeid_t>&      shortcuts() const { return shortcuts_; }
    // const OptionalCracks<TFlag>& treeFlags() const      { return overlayFlags_; }

private:
    size_t max_elements_;
    bitset_t treeSlots_;

    Cracks<nodeid_t>         additional_edges_;
    RefCracks<nodeid_t> shortcuts_;
    // OptionalCracks<TFlag> overlayFlags_;
    
    // Edge control limits and constants
    static constexpr size_t MAX_ADDITIONAL_EDGES_PER_NODE = 4;  // Upper limit per node for additional edges
    static constexpr size_t MAX_SHORTCUTS_PER_NODE = 12;         // Upper limit per node for shortcuts
    size_t additional_edges_limit_;
    size_t shortcuts_limit_;

    size_t allocateTree() {
        for (size_t i = 0; i < MAX_TREES; ++i) {
            if (!treeSlots_.test(i)) {
                treeSlots_.set(i);
                return i;
            }
        }
        return MAX_TREES;
    }

    // Edge reduction functions
    template<typename filter_t>
    void reduce_additional_edges(size_t target_reduction, const GraphStorage& graph, const GraphCompute<dist_t>& distf, const filter_t& filter);
    
    template<typename filter_t>
    void reduce_shortcuts(size_t target_reduction, const GraphStorage& graph, const GraphCompute<dist_t>& distf, const filter_t& filter);
    
    // Helper function for calculating neighbor similarity
    template<typename filter_t>
    int attribute_distance(nodeid_t node, nodeid_t neighbor, const GraphStorage& graph, const filter_t& filter) const;
};

} // namespace gaslib