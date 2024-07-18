#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

// No changes are needed in the C function as it already accepts pointers to the vectors.
float dotProduct(const float *a, size_t offsetB, size_t dims) {
    const float *b = (const float *)((const char *)a + offsetB);

    v128_t sum = wasm_f32x4_splat(0.0f);
    size_t i = 0;

    // Process in chunks of 4 using SIMD
    for (; i <= dims - 4; i += 4) {
        v128_t vecA = wasm_v128_load(a + i);
        v128_t vecB = wasm_v128_load(b + i);
        v128_t product = wasm_f32x4_mul(vecA, vecB);
        sum = wasm_f32x4_add(sum, product);
    }

    // Extract the results from the SIMD register and sum them
    float result[4];
    wasm_v128_store(result, sum);
    return result[0] + result[1] + result[2] + result[3];
}