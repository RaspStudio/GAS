#pragma once
#include <cstring>
#include <cassert>
#include "graph.h"

// Macro switch to enable/disable visited node tracking optimization
// Comment out this line to disable the optimization and use original O(n) approach
// #define ENABLE_VISITED_NODE_TRACKING

namespace gaslib {
using status_container_t = unsigned char;
using nodeid_t = unsigned int;

constexpr status_container_t STATUS_DISTED = 0x01;
constexpr status_container_t STATUS_FAILED = 0x02;
constexpr status_container_t STATUS_VISITED = 0x04;
constexpr status_container_t STATUS_FROM_SHORTCUT = 0x08;
constexpr nodeid_t INVALID_NODE_ID = 0xffffffff; // Invalid Node ID
constexpr size_t STATUS_DIRTY_MASK = 0x8000000000000000; // Mask to indicate dirty status
constexpr size_t FROMID_DIRTY_MASK = 0x4000000000000000; // Mask to indicate dirty status
static_assert(sizeof(STATUS_DIRTY_MASK) == sizeof(0x8000000000000000), "STATUS_DIRTY_MASK Byte Size Mismatch");

template <typename dist_t>
class StatusList {

    using dist_id_t = std::pair<dist_t, nodeid_t>;

private:
    size_t size_;
    
    struct Bitset {
        using WordType = uint64_t; 

        constexpr static unsigned int BITS_PER_WORD = sizeof(WordType) * 8;
        constexpr static unsigned int MASK = BITS_PER_WORD - 1;
        constexpr static unsigned int SHIFT = BITS_PER_WORD == 32 ? 5 : 6; // 2^5 = 32, 2^6 = 64
        static_assert(BITS_PER_WORD == 32 || BITS_PER_WORD == 64, "WordType must be 32 or 64 bits");

        std::vector<WordType> w;

        explicit Bitset(size_t n_bits) : w((n_bits + sizeof(WordType) * 8 - 1) / (sizeof(WordType) * 8), 0) {}

        inline bool test(nodeid_t i) const noexcept {
            return (w[i >> SHIFT] >> (i & MASK)) & static_cast<WordType>(1u);
        }
        inline void set(nodeid_t i) noexcept {
            w[i >> SHIFT] |= (static_cast<WordType>(1u) << (i & MASK));
        }
        inline void reset_all() noexcept {
            std::fill(w.begin(), w.end(), 0);
        }
        inline const void* ptr_for_prefetch(nodeid_t i) const noexcept {
            return &w[i >> SHIFT];
        }
    } visited_array_;

    struct NodeState {
        nodeid_t      from;   // 4B
        dist_t        dist;   // 4B
        nodeid_t      hop;    // 4B
        status_container_t      status; 
    };

    NodeState* nodes_;

public:

    std::vector<nodeid_t> result_array_;
    std::vector<nodeid_t> valid_array_;
#ifdef ENABLE_VISITED_NODE_TRACKING
    std::vector<nodeid_t> visited_nodes_; // Track visited node IDs for efficient stat_stages
#endif
    nodeid_t result_array_full_hop_ = 0; // Full Hop Count for Result Array
    nodeid_t max_hops_ = 0;
    nodeid_t l0_ep_ = INVALID_NODE_ID; // L0 Entry Point, used for L0 Search

    StatusList(size_t size) : size_(size),
                              visited_array_(size) {
        void* p = nullptr;
        constexpr size_t CACHE_LINE_SIZE = 64; 
        if (posix_memalign(&p, CACHE_LINE_SIZE, size * sizeof(NodeState)) != 0 || !p)
            throw std::bad_alloc();
        nodes_ = static_cast<NodeState*>(p);
        memset(nodes_, 0, size * sizeof(NodeState));
        for (size_t i = 0; i < size; ++i) {
            nodes_[i].from = INVALID_NODE_ID;
        }
        result_array_.reserve(200); // Reserve space for results TODO
    }

    ~StatusList() {
        if (nodes_) {
            free(nodes_);
            nodes_ = nullptr;
        }
    }

    inline size_t size() const {
        return size_ & (~STATUS_DIRTY_MASK & ~FROMID_DIRTY_MASK);
    }

    inline const void* ptr_for_prefetch(size_t index) const {
        assert(index < size());
        return &nodes_[index];
    }

    inline const void* ptr_for_prefetchbits(size_t index) const {
        assert(index < size());
        return visited_array_.ptr_for_prefetch(index);
    }

    inline void set_visitbit(nodeid_t index) {
        assert(index < size());
#ifdef ENABLE_VISITED_NODE_TRACKING
        // Only add to visited_nodes_ if not already visited
        if (!visited_array_.test(index)) {
            visited_nodes_.push_back(index);
        }
#endif
        // visited_array_[index] = true; // Set the visit bit
        visited_array_.set(index);
    }

    inline bool get_visitbit(nodeid_t index) const {
        assert(index < size());
        return visited_array_.test(index);
        // return visited_array_[index];
    }

    inline void set_status(size_t index, status_container_t status) {
        assert(index < size());
        nodes_[index].status |= status;
        // status_array_[index] |= status;
    }

    inline void set_from(size_t index, nodeid_t from, nodeid_t nhop) {
        assert(index < size());
        if (nodes_[index].from != INVALID_NODE_ID && nodes_[index].from != from) 
            throw std::runtime_error("From ID already set for index " + std::to_string(index));
        nodes_[index].from = from;
        nodes_[index].hop = nhop;
    }

    inline nodeid_t get_from(size_t index) const {
        assert(index < size());
        return nodes_[index].from;
    }

    inline nodeid_t get_hop(size_t index) const {
        assert(index < size());
        return nodes_[index].hop;
    }

    inline void set_dist(size_t index, dist_t dist) {
        assert(index < size());
        nodes_[index].dist = dist;
    }

    inline dist_t get_dist(size_t index) const {
        assert(index < size());
        return nodes_[index].dist;
    }

    inline bool has_status(size_t index, status_container_t status) const {
        assert(index < size());
        return (nodes_[index].status & status) != 0;
    }

    inline bool has_any_status(size_t index) const {
        assert(index < size());
        return nodes_[index].status != 0;
    }

    template <bool status_only = false>
    inline void clear_all() {
        // Clean Results
        if constexpr (!status_only) {
            // stat_visited_nodes();
            // Clean Visited
            size_ &= ~FROMID_DIRTY_MASK; // Clear dirty flag First!
            max_hops_ = 0;
            l0_ep_ = INVALID_NODE_ID; // Reset L0 Entry Point
            result_array_.clear();
            valid_array_.clear();
#ifdef ENABLE_VISITED_NODE_TRACKING
            visited_nodes_.clear(); // Clear visited nodes tracking
#endif
            result_array_full_hop_ = 0;
            assert(result_array_.size() == 0);
        }
        // Clean Status
        size_ &= ~STATUS_DIRTY_MASK;
        for (size_t i = 0; i < size(); ++i) {
            nodes_[i].status = 0;
            nodes_[i].from = INVALID_NODE_ID;
            nodes_[i].dist = 0;
            nodes_[i].hop = 0;
        }
        visited_array_.reset_all();
#ifdef ENABLE_VISITED_NODE_TRACKING
        visited_nodes_.clear(); // Clear visited nodes tracking
#endif
        // std::fill(visited_array_.begin(), visited_array_.end(), false); // Reset visited array
    }

    inline void set_dirty() {
        size_ |= STATUS_DIRTY_MASK;
    }

    inline void set_logable(size_t ef) {
        size_ |= FROMID_DIRTY_MASK;
        result_array_.reserve(ef);
    }

    inline void push_gas_results(nodeid_t id) {
        result_array_.push_back(id);
    }

    inline bool is_dirty() const {
        return (size_ & (STATUS_DIRTY_MASK | FROMID_DIRTY_MASK)) != 0;
    }


    void stat_visited_nodes() {
#ifdef ENABLE_VISITED_NODE_TRACKING
        for (const nodeid_t i : visited_nodes_) {
            if (has_status(i, STATUS_VISITED)) {
                printf("<%u>", i);
            }
        }
#else
        for (size_t i = 0; i < size(); ++i) {
            if (has_status(i, STATUS_VISITED)) {
                printf("<%lu>", i);
            }
        }
#endif
    }

    
    auto stat_stages(size_t k, const GraphStorage& graph) const -> std::vector<size_t> {
        
        constexpr nodeid_t ROOT_HOP = 1;
        if (max_hops_ == 0) {
            throw std::runtime_error("Max hops is not set, please run search first.");
        }
        
        std::vector<nodeid_t> visit_sequence;
        std::vector<nodeid_t> hop_to_dists;
        visit_sequence.resize(max_hops_ + 1, INVALID_NODE_ID);
        hop_to_dists.resize(max_hops_ + 1, 0);
#ifdef ENABLE_VISITED_NODE_TRACKING
        
        for (const nodeid_t i : visited_nodes_) {
            bool visited = get_visitbit(i);
            bool disted = has_status(i, STATUS_DISTED);
            
            if (disted) hop_to_dists[get_hop(i)]++;
            
            if (visited) {
                if (has_status(i, STATUS_FROM_SHORTCUT)) {
                    
                    continue;
                }
                
                nodeid_t i_hop = get_hop(i);
                nodeid_t from_id = get_from(i);
                
                
                if (from_id == i && has_status(i, STATUS_VISITED)) {
                    if (i_hop != ROOT_HOP) throw std::runtime_error("Root node hop mismatch! (i: " + std::to_string(i) + ", hop: " + std::to_string(i_hop) + ")");
                    
                    bool is_set = visit_sequence[ROOT_HOP] != INVALID_NODE_ID;
                    if (is_set) {
                        
                        if (visit_sequence[ROOT_HOP] != i) {
                            throw std::runtime_error("Root node already set to " + std::to_string(visit_sequence[ROOT_HOP]) + ", but trying to set to " + std::to_string(i));
                        }
                    } else {
                        
                        visit_sequence[ROOT_HOP] = i;
                    }
                }
                
                
                if (visit_sequence[i_hop] == INVALID_NODE_ID) {
                    
                    visit_sequence[i_hop] = from_id;
                } else if (visit_sequence[i_hop] != from_id) {
                    
                    throw std::runtime_error("From ID already set to " + std::to_string(visit_sequence[i_hop]) + ", but trying to set to " + std::to_string(from_id));
                }
            }
        }
#else
        
        size_t total_visited = 0;
        size_t size = this->size();
        for (size_t i = 0; i < size; ++i) {
            bool visited = get_visitbit(i);
            bool disted = has_status(i, STATUS_DISTED);
            
            if (visited) total_visited++;
            if (disted) hop_to_dists[get_hop(i)]++;
            
            if (visited) {
                if (has_status(i, STATUS_FROM_SHORTCUT)) {
                    
                    continue;
                }
                
                nodeid_t i_hop = get_hop(i);
                nodeid_t from_id = get_from(i);
                
                
                if (from_id == i && has_status(i, STATUS_VISITED)) {
                    if (i_hop != ROOT_HOP) throw std::runtime_error("Root node hop mismatch! (i: " + std::to_string(i) + ", hop: " + std::to_string(i_hop) + ")");
                    
                    bool is_set = visit_sequence[ROOT_HOP] != INVALID_NODE_ID;
                    if (is_set) {
                        
                        if (visit_sequence[ROOT_HOP] != i) {
                            throw std::runtime_error("Root node already set to " + std::to_string(visit_sequence[ROOT_HOP]) + ", but trying to set to " + std::to_string(i));
                        }
                    } else {
                        
                        visit_sequence[ROOT_HOP] = i;
                    }
                }
                
                
                if (visit_sequence[i_hop] == INVALID_NODE_ID) {
                    
                    visit_sequence[i_hop] = from_id;
                } else if (visit_sequence[i_hop] != from_id) {
                    
                    throw std::runtime_error("From ID already set to " + std::to_string(visit_sequence[i_hop]) + ", but trying to set to " + std::to_string(from_id));
                }
            }
        }
#endif

        
        using dist_id_t = std::pair<dist_t, nodeid_t>;

        struct CompareByFirst {
            inline bool operator()(dist_id_t const& a, dist_id_t const& b) const noexcept {
                return a.first < b.first;
            }
        };

        std::priority_queue<dist_id_t, std::vector<dist_id_t>, CompareByFirst> topk_heap;

        for (nodeid_t res : result_array_) {
            if (!has_status(res, STATUS_DISTED)) throw std::runtime_error("Result node without dist!");
            dist_t dist = get_dist(res);
            topk_heap.emplace(dist, res);
            if (topk_heap.size() > k) topk_heap.pop();
        }

        
        nodeid_t res_max_hop = 0;
        if (result_array_.empty()) {
            res_max_hop = max_hops_; 
        }
        while (!topk_heap.empty()) {
            nodeid_t res = topk_heap.top().second;
            nodeid_t res_hop = get_hop(res);
            if (res_hop > res_max_hop) res_max_hop = res_hop;
            topk_heap.pop();
        }

        
        std::vector<size_t> stages;
        stages.resize(6, 0); // 6 stages: stage1_hops, stage1_dists, stage2_hops, stage2_dists, stage3_hops, stage3_dists
        size_t& stage1_hops = stages[0];
        size_t& stage1_dists = stages[1];
        size_t& stage2_hops = stages[2];
        size_t& stage2_dists = stages[3];
        size_t& stage3_hops = stages[4];
        size_t& stage3_dists = stages[5];

        dist_t prev_dist = std::numeric_limits<dist_t>::max();
        bool in_stage1 = true;
        for (nodeid_t i = 1; i <= max_hops_; i++) {
            nodeid_t cur = visit_sequence[i];
            
            if (cur == INVALID_NODE_ID) {
                
                if (in_stage1) {
                    in_stage1 = false;
                    stage2_hops += 1;
                    stage2_dists += hop_to_dists[i];
                }
                else if (i <= res_max_hop) {
                    stage2_hops += 1;
                    stage2_dists += hop_to_dists[i];
                } else {
                    stage3_hops += 1;
                    stage3_dists += hop_to_dists[i];
                }
                continue;
            }
            
            if (i == ROOT_HOP) {
                prev_dist = std::numeric_limits<dist_t>::max();
                stage1_hops += 1;
                stage1_dists += hop_to_dists[i];
                continue;
            }
            if (!has_status(cur, STATUS_DISTED)) throw std::runtime_error("Hop without dist!");
            
            if (in_stage1) {
                
                dist_t cur_dist = get_dist(cur);
                if (cur_dist >= prev_dist) {
                    in_stage1 = false;
                    stage2_hops += 1;
                    stage2_dists += hop_to_dists[i];
                } else {
                    prev_dist = cur_dist;
                    stage1_hops += 1;
                    stage1_dists += hop_to_dists[i];
                }
            }
            
            else if (i <= res_max_hop) {
                stage2_hops += 1;
                stage2_dists += hop_to_dists[i];
            } 
            
            else {
                stage3_hops += 1;
                stage3_dists += hop_to_dists[i];
            }
        }

        if (stage1_hops + stage2_hops + stage3_hops != max_hops_) {
            throw std::runtime_error("Stage hops count mismatch! (stage1: " + std::to_string(stage1_hops) + \
                                     ", stage2: " + std::to_string(stage2_hops) + \
                                     ", stage3: " + std::to_string(stage3_hops) + \
                                     ", max_hops: " + std::to_string(max_hops_) + ")");
        }

        return stages;
    }

};

} // namespace gaslib