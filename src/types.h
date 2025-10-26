#pragma once
#include "../third-party/hnswlib/hnswlib.h"

namespace gaslib {

// Third-party library types
template <typename MTYPE>
using dist_func_t = ihnswlib::DISTFUNC<MTYPE>;
template <typename dist_t>
using AlgorithmInterface = ihnswlib::AlgorithmInterface<dist_t>;
template <typename dist_t>
using SpaceInterface = ihnswlib::SpaceInterface<dist_t>;

using VisitedList = ihnswlib::VisitedList;

using vl_type = ihnswlib::vl_type;
using labeltype = ihnswlib::labeltype;
using BaseFilterFunctor = ihnswlib::BaseFilterFunctor;
using linklistsizeint = unsigned int;



// Our own types
using nodelvl_t = unsigned char;


using label_t = size_t;
using nodeid_t = unsigned int;
using meta_t = int; 

using id_pair_t = std::pair<nodeid_t, nodeid_t>; 

using tableint = nodeid_t; 

using nodeid_t = unsigned int;
using gid_t = unsigned short;
using byte_t = char;

template <typename F, typename S>
struct ComparePairFirst {
    inline bool operator()(std::pair<F, S> const& a, std::pair<F, S> const& b) const noexcept {
        return a.first < b.first;
    }
};

} // namespace gaslib

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#if defined(USE_AVX512)
#define OPT_AVX512 1
#endif
#endif

namespace std {

static_assert((2 * sizeof(gaslib::nodeid_t)) == sizeof(size_t), 
              "gaslib::nodeid_t must be the half size as size_t for hash compatibility");
template <>
struct hash<std::pair<gaslib::nodeid_t, gaslib::nodeid_t>> {
    size_t operator()(const std::pair<gaslib::nodeid_t, gaslib::nodeid_t>& p) const {
        uint64_t high = static_cast<uint64_t>(p.first);
        uint64_t low = static_cast<uint64_t>(p.second);
        uint64_t combined = (high << 32) | low;

        combined ^= (combined >> 33);
        combined *= 0xff51afd7ed558ccdULL;
        combined ^= (combined >> 33);
        combined *= 0xc4ceb9fe1a85ec53ULL;
        combined ^= (combined >> 33);

        return static_cast<size_t>(combined);
    }
};

}