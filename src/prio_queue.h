#pragma once
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>
#include <stdexcept>
#include <concepts>

namespace gaslib {

template <typename H, typename V>
concept HeapConcept = 
requires { typename H::value_type; } && 
std::same_as<typename H::value_type, V> && 
requires(H heap, V v) {
    
    { heap.push(v) }      -> std::same_as<void>;
    { heap.push(std::move(v)) } -> std::same_as<void>;

    
    { heap.emplace(v) }   -> std::same_as<void>;
    { heap.emplace(std::move(v)) } -> std::same_as<void>;

    
    { heap.top() }        -> std::same_as<const V&>;

    
    { heap.pop() }        -> std::same_as<void>;

    
    { heap.empty() }      -> std::same_as<bool>;

    
    { heap.size() }       -> std::same_as<typename H::size_type>;
};


static_assert(HeapConcept<std::priority_queue<int>, int>);


template <typename T, typename Compare = std::less<T>>
class PriorityQueue {
private:
    Compare        comp_;

public:
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;

    std::vector<T> data_;

    
    explicit PriorityQueue(size_t reserve = 0) noexcept {
        data_.reserve(reserve + 1);
    }

    
    inline bool empty() const noexcept { return data_.empty(); }
    
    inline size_t size() const noexcept { return data_.size(); }
    
    inline void clear() noexcept { data_.clear(); }

    /// push lvalue
    inline void push(const T& value) noexcept {
        data_.push_back(value);
        std::push_heap(data_.begin(), data_.end(), comp_);
    }
    
    inline void push(T&& value) noexcept {
        data_.push_back(std::move(value));
        std::push_heap(data_.begin(), data_.end(), comp_);
    }
    
    template <typename... Args>
    inline void emplace(Args&&... args) noexcept {
        data_.emplace_back(std::forward<Args>(args)...);
        std::push_heap(data_.begin(), data_.end(), comp_);
    }

    
    inline const T& top() const noexcept {
        return data_.front();
    }

    
    inline void pop() noexcept {
        std::pop_heap(data_.begin(), data_.end(), comp_);
        data_.pop_back();
    }

    
    inline const T& visit_reverse(size_t idx) const noexcept {
        assert(idx < data_.size() && "Index out of bounds in visit_reverse()");
        return data_[data_.size() - 1 - idx];
    }

    
    
    inline void sort() noexcept {
        std::sort(data_.rbegin(), data_.rend(), comp_);
        assert(std::is_heap(data_.begin(), data_.end(), comp_) &&
               "Heap property violated after sort()");
    }

    inline T sort(size_t k) noexcept {
        std::nth_element(data_.rbegin(), data_.rbegin() + k, data_.rend(), comp_);
        // T ret = *(std::max_element(data_.rbegin(), data_.rbegin() + k, comp_));
        std::sort(data_.rbegin(), data_.rbegin() + k, comp_);
        T ret = data_[data_.size() - k];
        std::make_heap(data_.begin(), data_.end(), comp_);
        return ret;
    }

    void reserve(size_t cap) {
        data_.reserve(cap + 2);
    }
};

} // namespace gaslib
