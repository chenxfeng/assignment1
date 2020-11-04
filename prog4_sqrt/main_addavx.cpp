#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

extern void sqrt_avx_instrinsic(int N, float startGuess, float* values, float* output);

static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

int main() {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values; posix_memalign((void**)&values, 32, N*sizeof(float));///new float[N];
    float* output; posix_memalign((void**)&output, 32, N*sizeof(float));///new float[N];
    float* gold = new float[N];

    for (unsigned int i=0; i<N; i++)
    {
        // random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
        output[i] = 0.f;
    }
    // for (unsigned int i=0; i<N; i++)
    // {
    //     // // random input values
    //     // values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
    //     // TODO: Try different input values here.
    //     ///for best: limit the value range 0.5~1.5
    //     values[i] = .5f + 0.998f * static_cast<float>(rand()) / RAND_MAX;

    //     output[i] = 0.f;
    // }
    // for (unsigned int i=0; i<N; i+=4) {
    //     ///for worst: limit the value range 2.75~3
    //     values[i+0] = .5f + 0.998f * static_cast<float>(rand()) / RAND_MAX;
    //     values[i+1] = .5f + 0.998f * static_cast<float>(rand()) / RAND_MAX;
    //     values[i+2] = .5f + 0.998f * static_cast<float>(rand()) / RAND_MAX;
    //     values[i+3] = 2.751f + 0.248f * static_cast<float>(rand()) / RAND_MAX;

    //     output[i] = 0.f;
    //     output[i+1] = 0.f;
    //     output[i+2] = 0.f;
    //     output[i+3] = 0.f;
    // }

    // generate a gold version to check results
    for (unsigned int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the image using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 5; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    // verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;
    //
    // the AVX instrinsic code
    //
    double minInstrinsic = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_avx_instrinsic(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minInstrinsic = std::min(minInstrinsic, endTime - startTime);
    }
    printf("[sqrt avx instrinsic]:\t[%.3f] ms\n", minInstrinsic * 1000);
    verifyResult(N, output, gold);
    printf("\t\t\t\t(%.2fx speedup from AVX Instrinsic)\n", minSerial/minInstrinsic);

    free(values);///delete[] values;
    free(output);///delete[] output;
    delete[] gold;

    return 0;
}
