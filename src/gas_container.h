#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <bitset>
#include <span>
#include "types.h"

namespace gaslib {

constexpr size_t MAX_TREES = 256;
using bitset_t = std::bitset<MAX_TREES>;


template <typename T>
class Cracks {
public:
    Cracks(size_t max_elements) : data_(max_elements) {}

    // add value for node u
    void add(nodeid_t u, const T &val) {
        data_.at(u).push_back(val);
    }

    // clear all values for node u
    void clear(nodeid_t u) {
        data_.at(u).clear();
    }

    // remove specific value for node u
    void remove_force(nodeid_t u, const T &val) noexcept {
        auto &lst = data_.at(u);
        for (size_t i = 0; i < lst.size(); ++i) {
            if (lst[i] == val) {
                lst[i] = lst.back();
                lst.pop_back();
                return;
            }
        }
    }

    // get pointer to values for node u and set length
    const T* data(nodeid_t u) const {
        const auto &lst = data_.at(u);
        return lst.data();
    }

    // get size
    size_t size(nodeid_t u) const {
        return data_.at(u).size();
    }

    size_t count_all() const {
        size_t total = 0;
        for (const auto &lst : data_) {
            total += lst.size();
        }
        return total;
    }

private:
    std::vector<std::vector<T>> data_;
};


template <typename T>
class RefCracks {
public:
    struct Entry {
        T value;
        uint16_t refcount; 
    };

    RefCracks(size_t max_elements) : data_(max_elements) {}

    // add reference for node u
    void add(nodeid_t u, const T &val) noexcept {
        auto &lst = data_.at(u);
        for (auto &e : lst) {
            if (e.value == val) {
                ++e.refcount;
                return;
            }
        }
        lst.push_back({val, 1});
    }

    // remove reference for node u
    void remove(nodeid_t u, const T &val) noexcept {
        auto &lst = data_.at(u);
        for (size_t i = 0; i < lst.size(); ++i) {
            auto &e = lst[i];
            if (e.value == val) {
                if (e.refcount > 1) {
                    --e.refcount;
                } else {
                    lst[i] = lst.back();
                    lst.pop_back();
                }
                return;
            }
        }
    }

    void remove_forced(nodeid_t u, const T &val) noexcept {
        auto &lst = data_.at(u);
        for (size_t i = 0; i < lst.size(); ++i) {
            auto &e = lst[i];
            if (e.value == val) {
                lst[i] = lst.back();
                lst.pop_back();
                return;
            }
        }
    }

    // get pointer to entries for node u and set length
    const Entry* data(nodeid_t u) const noexcept {
        const auto &lst = data_.at(u);
        return lst.data();
    }

    const std::span<const Entry> span(nodeid_t u) const noexcept {
        return std::span<const Entry>(data_.at(u));
    }

    const size_t size(nodeid_t u) const noexcept {
        return data_.at(u).size();
    }

    const size_t count_all() const noexcept {
        size_t total = 0;
        for (const auto &lst : data_) {
            total += lst.size();
        }
        return total;
    }

private:
    std::vector<std::vector<Entry>> data_;
};


template <typename T>
class RefMutualCracks {
public:
    struct Entry {
        nodeid_t to;
        T value;
        uint16_t refcount;
    };

    explicit RefMutualCracks(size_t max_elements)
        : data_(max_elements) {}

    // add or increment undirected edge u<->v with associated value
    void add(nodeid_t u, nodeid_t v, const T &val) noexcept {
        // add/increment u->v
        add_dir(u, v, val);
        // add/increment v->u
        add_dir(v, u, val);
    }

    // remove or decrement undirected edge u<->v with associated value
    void remove(nodeid_t u, nodeid_t v, const T &val) noexcept {
        // remove/decrement u->v
        remove_dir(u, v, val);
        // remove/decrement v->u
        remove_dir(v, u, val);
    }

    // get pointer to directional entries for node u and set length
    const Entry* data(nodeid_t u) const noexcept {
        const auto &lst = data_.at(u);
        return lst.data();
    }

    // get span of directional entries for node u
    const std::span<const Entry> span(nodeid_t u) const noexcept {
        return std::span<const Entry>(data_.at(u));
    }

    // get size of directional entries for node u
    const size_t size(nodeid_t u) const noexcept {
        return data_.at(u).size();
    }

private:
    std::vector<std::vector<Entry>> data_;

    void add_dir(nodeid_t u, nodeid_t to, const T &val) noexcept {
        auto &lst = data_[u];
        for (auto &e : lst) {
            if (e.to == to && e.value == val) {
                ++e.refcount;
                return;
            }
        }
        lst.push_back({to, val, 1});
    }

    void remove_dir(nodeid_t u, nodeid_t to, const T &val) noexcept {
        auto &lst = data_[u];
        for (size_t i = 0; i < lst.size(); ++i) {
            auto &e = lst[i];
            if (e.to == to && e.value == val) {
                if (e.refcount > 1) {
                    --e.refcount;
                } else {
                    e = lst.back();
                    lst.pop_back();
                }
                return;
            }
        }
    }
};


template <typename T>
class OptionalCracks {
public:
    struct Entry {
        T value;
        bitset_t mask;
    };

    OptionalCracks(size_t max_elements) : data_(max_elements) {}

    // add value for node u under tree tid
    void add(nodeid_t u, const T &val, size_t tid) noexcept {
        auto &lst = data_.at(u);
        for (auto &e : lst) {
            if (e.value == val) {
                e.mask.set(tid);
                return;
            }
        }
        Entry e{val, {}};
        e.mask.set(tid);
        lst.push_back(std::move(e));
    }

    // remove value for node u under tree tid
    void remove(nodeid_t u, const T &val, size_t tid) noexcept {
        auto &lst = data_.at(u);
        for (size_t i = 0; i < lst.size(); ++i) {
            auto &e = lst[i];
            if (e.value == val) {
                e.mask.reset(tid);
                if (e.mask.none()) {
                    lst[i] = lst.back();
                    lst.pop_back();
                }
                return;
            }
        }
    }

    // get pointer to entries for node u and set length
    // caller can iterate entries[0..len)
    const Entry* data(nodeid_t u) const noexcept {
        const auto &lst = data_.at(u);
        return lst.data();
    }

    // get span of entries for node u
    const std::span<const Entry> span(nodeid_t u) const noexcept {
        return std::span<const Entry>(data_.at(u));
    }

    // get size of entries for node u
    const size_t size(nodeid_t u) const noexcept {
        return data_.at(u).size();
    }

    // get total count of all entries
    size_t count_all() const noexcept {
        size_t total = 0;
        for (const auto &lst : data_) {
            total += lst.size();
        }
        return total;
    }

private:
    std::vector<std::vector<Entry>> data_;
};

}