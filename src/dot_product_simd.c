#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

float dotProduct(const float *a, const float *b, size_t dims) {
    v128_t sum = wasm_f32x4_splat(0.0f);

    // process in chunks of 4, memory/instruction alignment
    for (size_t i = 0; i < dims; i += 4) {
        v128_t vecA = wasm_v128_load(&a[i]);
        v128_t vecB = wasm_v128_load(&b[i]);
        v128_t product = wasm_f32x4_mul(vecA, vecB);
        sum = wasm_f32x4_add(sum, product);
    }

    // extract from the SIMD register and sum up
    float result[4];
    wasm_v128_store(result, sum);
    return result[0] + result[1] + result[2] + result[3];
}