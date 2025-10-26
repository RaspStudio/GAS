#pragma once
#include "types.h"
#include "gas_interface.h"

namespace gaslib {

template <size_t adedge_ver = 0, size_t adedge_size = 3, size_t shortcut_ver = 0, size_t shortcut_size = 3, size_t triprune_ver = 0>
struct GASTreeBuilder {

    template <typename dist_t, typename filter_t>
    static GASTree<dist_t, filter_t> build(
        const StatusList<dist_t>& status_list,
        const GraphStorage& graph,
        const GASStorage<dist_t>& cgraph,
        const GraphCompute<dist_t>& distf,
        const filter_t& filter);
};

} // namespace gaslib