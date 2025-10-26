#pragma once

#include <set>
#include <utility>
#include <variant>
#include <algorithm>
#include <cmath>
#include "types.h"
#include <concepts>

namespace gaslib {


static_assert(sizeof(size_t) == sizeof(meta_t) + sizeof(nodeid_t), "Must use 64-bit size_t for label + id");

class GASFilterFunctor : public BaseFilterFunctor {
public:
    unsigned char type_; 
    constexpr static unsigned char TYPE_RANGE = 1; 
    constexpr static unsigned char TYPE_TAG = 2; 

    GASFilterFunctor(unsigned char type) : type_(type) {} 

    
    virtual bool operator()(size_t label) const noexcept override = 0;

    
    virtual int similarity(const GASFilterFunctor& other) const noexcept = 0;
};

class RangeGASFilterFunctor : public GASFilterFunctor {
public:
    meta_t start; 
    meta_t end;   
    
    RangeGASFilterFunctor(meta_t start, meta_t end) : GASFilterFunctor(TYPE_RANGE), start(start), end(end) {
        if (start > end) {
            std::cerr << "Invalid range: start (" << start << ") is greater than end (" << end << ")" << std::endl;
            throw std::invalid_argument("Start must be less than or equal to end");
        }
    }

    inline bool operator()(size_t label) const noexcept override {
        meta_t meta = label >> 32; 
        return start <= meta && meta <= end;
    }

    inline int similarity(const GASFilterFunctor& other) const noexcept override {
        if (other.type_ == TYPE_RANGE) {
            auto* rangeFilter = reinterpret_cast<RangeGASFilterFunctor*>(const_cast<GASFilterFunctor*>(&other));
            
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

class TagGASFilterFunctor : public GASFilterFunctor {
public:
    std::set<meta_t> tags; 
    
    TagGASFilterFunctor(const std::set<meta_t>& tags) : GASFilterFunctor(TYPE_TAG), tags(tags) {
        if (tags.empty()) {
            throw std::invalid_argument("Tags set cannot be empty");
        }
    }

    inline bool operator()(size_t label) const noexcept override {
        meta_t meta = label >> 32; 
        return tags.find(meta) != tags.end(); 
    }

    inline int similarity(const GASFilterFunctor& other) const noexcept override {
        if (other.type_ == TYPE_TAG) {
            auto* otherf = reinterpret_cast<TagGASFilterFunctor*>(const_cast<GASFilterFunctor*>(&other));
            
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

    WrappedFilterFunctor(const GASFilterFunctor* filter, std::vector<label_t> labels)
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

template <typename T>
concept FilterConcept = requires(const T* f, label_t label) {
    { (*f)(label) } noexcept -> std::convertible_to<bool>;       
};

template <typename T>
concept GASFilterConcept = requires(const T* f, const GASFilterFunctor* cf, label_t label) {
    { (*f)(label) } noexcept -> std::convertible_to<bool>;       
    { (*f).similarity(*cf) } noexcept -> std::same_as<int>;         
};

} // namespace gaslib