#pragma once
#include "types.h"
#include "graph.h"

namespace gaslib {

template<typename dist_t> 
class ShortcutStorage;

template<typename dist_t, typename filter_t> 
requires GasFilterConcept<filter_t>
struct GasTree;

template<typename dist_t> 
class StatusList;

template <size_t adedge_ver = 0, size_t adedge_size = 3, size_t shortcut_ver = 0, size_t shortcut_size = 3, size_t triprune_ver = 0>
struct GasTreeBuilder {

    template <typename dist_t, typename filter_t, GraphConcept graph_t>
    static GasTree<dist_t, filter_t> build(
        const StatusList<dist_t>& status_list,
        const graph_t& graph,
        const ShortcutStorage<dist_t>& cgraph,
        const GraphCompute<dist_t>& distf,
        const filter_t& filter);
};

} // namespace gaslib
