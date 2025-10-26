#pragma once
#include "hnswlib/hnswlib.h"

namespace gaslib {

#if defined(OPT_AVX512)
static inline float
L2SqrSIMD16ExtAVX512_opt(const void* pVect1v, const void* pVect2v, const void* qty_ptr) noexcept {
    const size_t qty = *(const size_t*)qty_ptr;

    
#if defined(__GNUC__) || defined(__clang__)
    const float* __restrict a = (const float*)__builtin_assume_aligned(pVect1v, 64);
    const float* __restrict b = (const float*)__builtin_assume_aligned(pVect2v, 64);
#else
    const float* __restrict a = (const float*)pVect1v;
    const float* __restrict b = (const float*)pVect2v;
#endif

    size_t i = 0;
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    
    for (; i + 63 < qty; i += 64) {
        
        __m512 a0 = _mm512_load_ps(a + i);
        __m512 b0 = _mm512_load_ps(b + i);
        __m512 a1 = _mm512_load_ps(a + i + 16);
        __m512 b1 = _mm512_load_ps(b + i + 16);
        __m512 a2 = _mm512_load_ps(a + i + 32);
        __m512 b2 = _mm512_load_ps(b + i + 32);
        __m512 a3 = _mm512_load_ps(a + i + 48);
        __m512 b3 = _mm512_load_ps(b + i + 48);

        __m512 d0 = _mm512_sub_ps(a0, b0);
        __m512 d1 = _mm512_sub_ps(a1, b1);
        __m512 d2 = _mm512_sub_ps(a2, b2);
        __m512 d3 = _mm512_sub_ps(a3, b3);

        sum0 = _mm512_fmadd_ps(d0, d0, sum0);
        sum1 = _mm512_fmadd_ps(d1, d1, sum1);
        sum2 = _mm512_fmadd_ps(d2, d2, sum2);
        sum3 = _mm512_fmadd_ps(d3, d3, sum3);
    }

    
    for (; i + 15 < qty; i += 16) {
        __m512 av = _mm512_load_ps(a + i);
        __m512 bv = _mm512_load_ps(b + i);
        __m512 d  = _mm512_sub_ps(av, bv);
        sum0 = _mm512_fmadd_ps(d, d, sum0);
    }

    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));

#if defined(__GNUC__) || defined(__clang__)
    return _mm512_reduce_add_ps(sum);
#else
    __m256 lo256 = _mm512_castps512_ps256(sum);
    __m256 hi256 = _mm512_extractf32x8_ps(sum, 1);
    __m256 s256  = _mm256_add_ps(lo256, hi256);
    __m128 lo128 = _mm256_castps256_ps128(s256);
    __m128 hi128 = _mm256_extractf128_ps(s256, 1);
    __m128 s128  = _mm_add_ps(lo128, hi128);
    __m128 shuf  = _mm_movehdup_ps(s128);
    s128         = _mm_add_ps(s128, shuf);
    shuf         = _mm_movehl_ps(shuf, s128);
    s128         = _mm_add_ss(s128, shuf);
    return _mm_cvtss_f32(s128);
#endif
}
#endif

class L2SpaceOptAVX512Dim16 : public ihnswlib::SpaceInterface<float> {
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceOptAVX512Dim16(size_t dim) {
#if defined(OPT_AVX512)
        if (!AVX512Capable() || dim % 16 != 0) 
            throw std::runtime_error("AVX512 is not supported in this build or dimension is not a multiple of 16");
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    ihnswlib::DISTFUNC<float> get_dist_func() {
#if defined(OPT_AVX512)
        return L2SqrSIMD16ExtAVX512_opt;
#else
        throw std::runtime_error("AVX512 is not supported in this build");
#endif
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceOptAVX512Dim16() {}
};

} // namespace ihnswlib