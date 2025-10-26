#pragma once
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <span>
#include "types.h"
#include "opt.h"

namespace gaslib {


class GraphStorage {
public:
    size_t max_elements_{ 0 };
    size_t cur_elements_{ 0 };

private:
    size_t size_rawvector_{ 0 };
    size_t max_links_per_element0_{ 0 };
    size_t max_links_per_element1_{ 0 };
    size_t size_neighbor_list0_{ 0 };
    size_t size_neighbor_list1_{ 0 };
    size_t size_per_element0_{ 0 };

    
    size_t off_vec_mem_{ 0 };        
    size_t off_label_mem_{ 0 };      
    size_t off_neighbors_mem_{ 0 };  
    size_t stride0_mem_{ 0 };        

    char* memory_l0_{ nullptr };
    char** memory_l1_{ nullptr };
    std::vector<char> memory_levels_;

    
    static constexpr size_t align_up(size_t x, size_t a) noexcept { return (x + (a - 1)) & ~(a - 1); }
 

public:

    struct neighbor_list {
        nodeid_t len_;
        nodeid_t neighbor_[];

        std::span<nodeid_t> span() {
            return std::span<nodeid_t>(neighbor_, len_);
        }

        inline bool contains(nodeid_t nbr) {
            for (nodeid_t i = 0; i < len_; ++i) {
                if (neighbor_[i] == nbr) {
                    return true;
                }
            }
            return false;
        }
    };


    GraphStorage(size_t max_elements, size_t size_rawvector, size_t max_links_per_element) 
        : max_elements_(max_elements),
        size_rawvector_(size_rawvector),
        max_links_per_element0_(max_links_per_element * 2),
        max_links_per_element1_(max_links_per_element),
        memory_levels_(max_elements) {
        size_neighbor_list0_ = sizeof(neighbor_list) + sizeof(nodeid_t) * max_links_per_element0_;
        size_neighbor_list1_ = sizeof(neighbor_list) + sizeof(nodeid_t) * max_links_per_element1_;

        
        size_per_element0_ = size_rawvector + sizeof(label_t) + size_neighbor_list0_;

        
        off_vec_mem_       = 0;                                                          
        size_t after_vec   = size_rawvector_;
        off_label_mem_     = align_up(after_vec, 8);
        size_t after_label = off_label_mem_ + sizeof(label_t);
        off_neighbors_mem_ = align_up(after_label, 4);
        stride0_mem_       = align_up(off_neighbors_mem_ + size_neighbor_list0_, 64);    

        
        void* p0 = nullptr;
        if (posix_memalign(&p0, 64, stride0_mem_ * max_elements_))
            throw std::runtime_error("Not enough memory: GraphStorage failed to allocate memory");
        memory_l0_ = (char*)p0;
        
        memset(memory_l0_, 0, stride0_mem_ * max_elements_);

        
        memory_l1_ = (char**) malloc(sizeof(char*) * max_elements);
        if (memory_l1_ == nullptr)
            throw std::runtime_error("Not enough memory: GraphStorage failed to allocate memory for link lists");
        memset(memory_l1_, 0, sizeof(char*) * max_elements);
    }

    ~GraphStorage() {
        free(memory_l0_);
        for (size_t i = 0; i < max_elements_; i++) {
            if (memory_levels_[i] > 0) {
                if (memory_l1_[i] == nullptr) {
                    std::cerr << "Warning: memory_l1_[" << i << "] is null, but level is " << (int)memory_levels_[i] << std::endl;
                    exit(EXIT_FAILURE);
                }
                free(memory_l1_[i]);
                memory_levels_[i] = 0;
            }
        }
        free(memory_l1_);
    }

    constexpr char* get_data_ptr(tableint internal_id) const {
        return memory_l0_ + internal_id * stride0_mem_ + off_vec_mem_;
    }

    label_t* get_label_ptr(tableint internal_id) const {
        return (label_t*) (memory_l0_ + internal_id * stride0_mem_ + off_label_mem_);
    }

    neighbor_list* get_neighbors_l0(tableint internal_id) const {
        return (neighbor_list*) (memory_l0_ + internal_id * stride0_mem_ + off_neighbors_mem_);
    }

    neighbor_list* get_neighbors_l1(tableint internal_id, nodelvl_t level) const {
        assert(level > 0);
        return (neighbor_list*) (memory_l1_[internal_id] + (level - 1) * size_neighbor_list1_);
    }

    nodeid_t init_storage(nodelvl_t level) {
        if (cur_elements_ >= max_elements_) {
            throw std::runtime_error("GraphStorage: Cannot add more elements, storage is full");
        }
        nodeid_t internal_id = cur_elements_;
        cur_elements_++;
        memset(memory_l0_ + internal_id * stride0_mem_, 0, stride0_mem_);
        memory_levels_[internal_id] = level;
        if (level > 0) {
            memory_l1_[internal_id] = (char*) malloc(size_neighbor_list1_ * level);
            if (memory_l1_[internal_id] == nullptr)
                throw std::runtime_error("Not enough memory: GraphStorage failed to allocate memory for link lists");
            memset(memory_l1_[internal_id], 0, size_neighbor_list1_ * level);
        } else {
            memory_l1_[internal_id] = nullptr;
        }
        return internal_id;
    }

    nodelvl_t level(tableint internal_id) const {
        assert(internal_id < max_elements_);
        return memory_levels_[internal_id];
    }

    void resize(size_t max_elements) {
        if (max_elements < max_elements_)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        char* memory_l0_new = nullptr;
        if (posix_memalign((void**)(&memory_l0_new), 64, stride0_mem_ * max_elements))
            throw std::runtime_error("Not enough memory: GraphStorage failed to allocate memory");

        
        for (size_t i = 0; i < max_elements_; ++i) {
            memcpy(memory_l0_new + i * stride0_mem_,
                   memory_l0_    + i * stride0_mem_,
                   stride0_mem_);
        }

        
        if (max_elements > max_elements_) {
            memset(memory_l0_new + max_elements_ * stride0_mem_, 0,
                   (max_elements - max_elements_) * stride0_mem_);
        }

        char** memory_l1_new = (char**) realloc(memory_l1_, max_elements * sizeof(char*));
        if (memory_l1_new == nullptr)
            throw std::runtime_error("Not enough memory: GraphStorage failed to allocate memory for link lists");

        
        free(memory_l0_);
        memory_l0_ = memory_l0_new;
        memory_l1_ = memory_l1_new;
        memory_levels_.resize(max_elements, 0);
        max_elements_ = max_elements;
    }

    bool save(std::ofstream& ofs) {
        size_t store_magic_number = 0x12345678;
        ofs.write((const char*)(&store_magic_number), sizeof(store_magic_number));
        
        
        for (size_t i = 0; i < max_elements_; i++) {
            ofs.write((const char*)get_neighbors_l0(i), size_neighbor_list0_);
            ofs.write((const char*)get_label_ptr(i), sizeof(label_t));
            ofs.write(get_data_ptr(i), size_rawvector_);
        }
        
        ofs.write((const char*)(&store_magic_number), sizeof(store_magic_number));
        ofs.write((const char*)memory_levels_.data(), max_elements_ * sizeof(char));
        for (size_t i = 0; i < max_elements_; i++) {
            if (memory_levels_[i] > 0) {
                ofs.write(memory_l1_[i], size_neighbor_list1_ * memory_levels_[i]);
            }
            size_t mid_magic_number = 0x12343444;
            ofs.write((const char*)(&mid_magic_number), sizeof(mid_magic_number));
        }
        size_t final_magic_number = 0xAACCDDEE;
        ofs.write((const char*)(&final_magic_number), sizeof(final_magic_number));
        return true;
    }

    bool load(std::ifstream& ifs) {
        size_t store_magic_number;
        ifs.read((char*)(&store_magic_number), sizeof(store_magic_number));
        if (store_magic_number != 0x12345678) {
            throw std::runtime_error("GraphStorage: Invalid magic number in load");
        }
        
        
        for (size_t i = 0; i < max_elements_; i++) {
            ifs.read((char*)get_neighbors_l0(i), size_neighbor_list0_);
            ifs.read((char*)get_label_ptr(i), sizeof(label_t));
            ifs.read(get_data_ptr(i), size_rawvector_);
        }

        store_magic_number = 0;
        ifs.read((char*)(&store_magic_number), sizeof(store_magic_number));
        if (store_magic_number != 0x12345678) {
            throw std::runtime_error("GraphStorage: Invalid magic number in load");
        }

        ifs.read((char*)memory_levels_.data(), max_elements_ * sizeof(char));
        cur_elements_ = max_elements_;
        for (size_t i = 0; i < max_elements_; i++) {
            
            if (memory_l1_[i] != nullptr) {
                throw std::runtime_error("GraphStorage: L1 memory pointer is not null before load");
            }
            if (memory_levels_[i] > 0) {
                memory_l1_[i] = (char*) malloc(size_neighbor_list1_ * memory_levels_[i]);
                if (memory_l1_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: GraphStorage failed to allocate memory for link lists");
                ifs.read(memory_l1_[i], size_neighbor_list1_ * memory_levels_[i]);
            } else {
                memory_l1_[i] = nullptr;
            }
            size_t mid_magic_number;
            ifs.read((char*)(&mid_magic_number), sizeof(mid_magic_number));
            if (mid_magic_number != 0x12343444) {
                throw std::runtime_error("GraphStorage: Invalid magic number in load");
            }
        }
        
        size_t final_magic_number;
        ifs.read((char*)(&final_magic_number), sizeof(final_magic_number));
        if (final_magic_number != 0xAACCDDEE) {
            throw std::runtime_error("GraphStorage: Invalid final magic number in load");
        }
        return true;
    }
};

template <typename dist_t>
class GraphCompute {
public:
    const dist_func_t<dist_t> fstdistfunc_;
    const void* dist_func_param_;

    GraphCompute(dist_func_t<dist_t> fstdistfunc, void* dist_func_param)
        : fstdistfunc_(fstdistfunc), dist_func_param_(dist_func_param) {}

    inline dist_t get_distance(const void* a, const void* b) const {
        return fstdistfunc_(a, b, dist_func_param_);
    }
};

template <>
class GraphCompute<float> {
    const bool use_avx512_;

public:
    const dist_func_t<float> fstdistfunc_;
    const void* dist_func_param_;

    GraphCompute(dist_func_t<float> fstdistfunc, void* dist_func_param)
        : use_avx512_(
#if defined(OPT_AVX512)
            fstdistfunc == L2SqrSIMD16ExtAVX512_opt
#else
            false
#endif
            ), fstdistfunc_(fstdistfunc), dist_func_param_(dist_func_param) {}

    inline float get_distance(const void* a, const void* b) const {
        if (use_avx512_) {
#ifdef USE_AVX512
            return L2SqrSIMD16ExtAVX512_opt(a, b, dist_func_param_);
#else
            throw std::runtime_error("AVX512 is not supported in this build");
#endif
        } else {
            return fstdistfunc_(a, b, dist_func_param_);
        }
    }
};


}; // namespace gaslib