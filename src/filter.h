#pragma once

#include <set>
#include <utility>
#include <algorithm>
#include <cmath>
#include "types.h"
#include <concepts>

namespace gaslib {


static_assert(sizeof(size_t) == sizeof(meta_t) + sizeof(nodeid_t), "Must use 64-bit size_t for label + id");

class GasFilterFunctor : public BaseFilterFunctor {
public:
    unsigned char type_;
    constexpr static unsigned char TYPE_RANGE = 1;
    constexpr static unsigned char TYPE_TAG = 2;

    GasFilterFunctor(unsigned char type) : type_(type) {}

    virtual bool operator()(size_t label) const noexcept override = 0;

    virtual int similarity(const GasFilterFunctor& other) const noexcept = 0;
};

class RangeGasFilterFunctor : public GasFilterFunctor {
public:
    meta_t start;
    meta_t end;
    
    RangeGasFilterFunctor(meta_t start, meta_t end) : GasFilterFunctor(TYPE_RANGE), start(start), end(end) {
        if (start > end) {
            std::cerr << "Invalid range: start (" << start << ") is greater than end (" << end << ")" << std::endl;
            throw std::invalid_argument("Start must be less than or equal to end");
        }
    }

    inline bool operator()(size_t label) const noexcept override {
        meta_t meta = label >> 32;
        return start <= meta && meta <= end;
    }

    inline int similarity(const GasFilterFunctor& other) const noexcept override {
        if (other.type_ == TYPE_RANGE) {
            auto* rangeFilter = reinterpret_cast<RangeGasFilterFunctor*>(const_cast<GasFilterFunctor*>(&other));
            size_t intersectionStart = std::max(start, rangeFilter->start);
            size_t intersectionEnd = std::min(end, rangeFilter->end);
            if (intersectionStart <= intersectionEnd) {
                size_t intersectionSize = intersectionEnd - intersectionStart + 1;
                size_t currentSize = end - start + 1;
                return static_cast<int>(std::round((static_cast<double>(intersectionSize) / currentSize) * 100));
            }
        }
        return 0;
    }
};

class TagGasFilterFunctor : public GasFilterFunctor {
public:
    std::set<meta_t> tags;
    
    TagGasFilterFunctor(const std::set<meta_t>& tags) : GasFilterFunctor(TYPE_TAG), tags(tags) {
        if (tags.empty()) {
            throw std::invalid_argument("Tags set cannot be empty");
        }
    }

    inline bool operator()(size_t label) const noexcept override {
        meta_t meta = label >> 32;
        return tags.find(meta) != tags.end();
    }

    inline int similarity(const GasFilterFunctor& other) const noexcept override {
        if (other.type_ == TYPE_TAG) {
            auto* otherf = reinterpret_cast<TagGasFilterFunctor*>(const_cast<GasFilterFunctor*>(&other));
            size_t intersectionCount = 0;
            for (const auto& tag : tags) {
                if (otherf->tags.find(tag) != otherf->tags.end()) {
                    intersectionCount++;
                }
            }
            return static_cast<int>(std::round((static_cast<double>(intersectionCount) / tags.size()) * 100));
        }
        return 0;
    }
};


class WrappedFilterFunctor : public BaseFilterFunctor {
public:
    const BaseFilterFunctor* filter_;
    std::vector<label_t> labels_;

    WrappedFilterFunctor(const GasFilterFunctor* filter, std::vector<label_t> labels)
        : filter_(filter), labels_(std::move(labels)) {}

    inline bool operator()(size_t label) const noexcept override {
        nodeid_t id = label & 0xFFFFFFFF;
        if (id >= labels_.size()) {
            std::cerr << "Invalid label: " << label << ", exceeds labels size: " << labels_.size() << std::endl;
            return false;
        }
        label_t node_label = labels_[id];
        return (*filter_)(node_label);
    }
};


class MultiWrappedFilterFunctor : public WrappedFilterFunctor {
public:
    MultiWrappedFilterFunctor(std::vector<const BaseFilterFunctor*> filters,
                             std::vector<const std::vector<label_t>*> labels_list)
        : WrappedFilterFunctor(nullptr, {}), filters_(std::move(filters)), labels_list_(std::move(labels_list)) {
        if (filters_.size() != labels_list_.size()) {
            throw std::invalid_argument("MultiWrappedFilterFunctor: filters/labels_list size mismatch");
        }
    }

    inline bool operator()(size_t label) const noexcept override {
        nodeid_t id = label & 0xFFFFFFFF;

        for (size_t i = 0; i < filters_.size(); ++i) {
            const auto* f = filters_[i];
            const auto* labels = labels_list_[i];
            if (f == nullptr) return false;
            if (labels == nullptr) return false;
            if (id >= labels->size()) return false;
            if (!(*f)((*labels)[id])) return false;
        }
        return true;
    }

    std::vector<const BaseFilterFunctor*> filters_;
    std::vector<const std::vector<label_t>*> labels_list_;
};

template <typename PrimaryFilterT, typename SecondaryFilterT>
class ComposedFilterFunctor : public PrimaryFilterT {
public:
    using primary_filter_type = PrimaryFilterT;
    using secondary_filter_type = SecondaryFilterT;

    ComposedFilterFunctor(const PrimaryFilterT& primary,
                          const SecondaryFilterT& secondary)
        : PrimaryFilterT(primary), secondary_(secondary) {}

    inline const primary_filter_type& primary() const noexcept { return *this; }
    inline const secondary_filter_type& secondary() const noexcept { return secondary_; }

private:
    SecondaryFilterT secondary_;
};

template <typename T>
struct composed_filter_traits {
    static constexpr bool value = false;
};

template <typename P, typename S>
struct composed_filter_traits<ComposedFilterFunctor<P, S>> {
    static constexpr bool value = true;
    using primary = P;
    using secondary = S;
};

template <typename T>
inline constexpr bool is_composed_filter_v = composed_filter_traits<T>::value;


template <typename T>
concept FilterConcept = requires(const T* f, label_t label) {
    { (*f)(label) } noexcept -> std::convertible_to<bool>;
};

template <typename T>
concept GasFilterConcept = requires(const T* f, const GasFilterFunctor* cf, label_t label) {
    { (*f)(label) } noexcept -> std::convertible_to<bool>;
    { (*f).similarity(*cf) } noexcept -> std::same_as<int>;
};

} // namespace gaslib