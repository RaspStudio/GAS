#pragma once
#include "status_list.h"
#include <deque>
#include <mutex>
#include <condition_variable>

namespace gaslib {

template <typename dist_t>
class StatusListPool {
public:
    enum class State {
        CLEARED,
        SEARCHED_UNCOUNTED,
        SEARCHED_COUNTED,
        SEARCHED_CONSUMED
    };

private:
    std::deque<StatusList<dist_t>*> pool_[4];
    mutable std::mutex pool_guard_;
    std::condition_variable pool_cv_; 
    size_t size_;

public:
    StatusListPool(size_t init_pool_size, size_t size) : size_(size) {
        for (size_t i = 0; i < init_pool_size; i++) {
            pool_[static_cast<int>(State::CLEARED)].push_back(new StatusList<dist_t>(size));
        }
    }

    /**
     * Acquire a StatusList in the specified state; block if none available.
     */
    StatusList<dist_t>* acquire(State required_state) {
        std::unique_lock<std::mutex> lock(pool_guard_);
        
        int state_idx = static_cast<int>(required_state);
        
        pool_cv_.wait(lock, [this, state_idx] { return !pool_[state_idx].empty(); });

        StatusList<dist_t>* status_list = pool_[state_idx].front();
        pool_[state_idx].pop_front();
        
        return status_list;
    }

    /**
     * Return a StatusList to the pool for the specified state.
     */
    void release(StatusList<dist_t>* status_list, State new_state) {
        {
            std::unique_lock<std::mutex> lock(pool_guard_);
            int state_idx = static_cast<int>(new_state);
            pool_[state_idx].push_back(status_list);
        }
        pool_cv_.notify_one();
    }

    /**
     * Get the current size of the pool for the specified state.
     */
    size_t pool_size(State state) const {
        std::unique_lock<std::mutex> lock(pool_guard_);
        int state_idx = static_cast<int>(state);
        return pool_[state_idx].size();
    }

    /**
     * Get the total size across all pools.
     */
    size_t total_pool_size() const {
        std::unique_lock<std::mutex> lock(pool_guard_);
        return pool_[0].size() + pool_[1].size() + pool_[2].size() + pool_[3].size();
    }

    ~StatusListPool() {
        std::unique_lock<std::mutex> lock(pool_guard_);
        for (int i = 0; i < 4; i++) {
            while (!pool_[i].empty()) {
                StatusList<dist_t>* status_list = pool_[i].front();
                pool_[i].pop_front();
                delete status_list;
            }
        }
    }

    StatusListPool(const StatusListPool&) = delete;
    StatusListPool& operator=(const StatusListPool&) = delete;
};

} // namespace gaslib
