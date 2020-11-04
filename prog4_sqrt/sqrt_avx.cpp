#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void sqrt_avx_instrinsic(int N,
                float initialGuess,
                float values[],
                float output[])
{
    // static const float kThreshold = 0.00001f;
    __m256 kThreshold_p = _mm256_set1_ps(0.00001f);
    __m256 kThreshold_n = _mm256_set1_ps(-0.00001f);
    __m256 v_one = _mm256_set1_ps(1.0f);

    // for (int i=0; i<N; i++) {
    for (int i = 0; i < N; i += 8) {//assume N divisible by Vector width 8

        // float x = values[i];
        __m256 x = _mm256_load_ps(&(values[i]));
        // float guess = initialGuess;
        __m256 guess = _mm256_set1_ps(initialGuess);

        // float error = fabs(guess * guess * x - 1.f);
        __m256 error = _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x), v_one);

        // while (error > kThreshold) {
        //     guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
        //     error = fabs(guess * guess * x - 1.f);
        // }
        ///error <= kThreshold && error >= -kThreshold) : break
        __m256 cmp1 = _mm256_cmp_ps(error, kThreshold_p, _CMP_GT_OQ);
        __m256 cmp2 = _mm256_cmp_ps(error, kThreshold_n, _CMP_LT_OQ);
        cmp1 = _mm256_or_ps(cmp1, cmp2);
        unsigned char mask = _mm256_movemask_ps(cmp1) & 255;
        while (mask != 0) {
            guess = _mm256_blendv_ps(guess, 
                _mm256_mul_ps(_mm256_set1_ps(0.5f), 
                _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(3.f), guess),
                _mm256_mul_ps(_mm256_mul_ps(x, guess), 
                _mm256_mul_ps(guess, guess)))), cmp1);
            error = _mm256_blendv_ps(error, 
                _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x), 
                    v_one), cmp1);
            cmp1 = _mm256_cmp_ps(error, kThreshold_p, _CMP_GT_OQ);
            cmp2 = _mm256_cmp_ps(error, kThreshold_n, _CMP_LT_OQ);
            cmp1 = _mm256_or_ps(cmp1, cmp2);
            mask = _mm256_movemask_ps(cmp1) & 255;
        }

        // output[i] = x * guess;
        _mm256_store_ps(&(output[i]), _mm256_mul_ps(x, guess));
    }
}

