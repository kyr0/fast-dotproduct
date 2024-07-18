// compile: bun run build:wasm
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

void dotProduct(const float *data, size_t numVectors, size_t dimsPerVector, float *results) {
    size_t stride = dimsPerVector * 2; // Two vectors per pair (A and B)

    for (size_t vectorIndex = 0; vectorIndex < numVectors; ++vectorIndex) {
        v128_t sum = wasm_f32x4_splat(0.0f);
        size_t offsetA = vectorIndex * stride;
        size_t offsetB = offsetA + dimsPerVector;

        for (size_t i = 0; i < dimsPerVector; i += 4) {
            v128_t vecA = wasm_v128_load(data + offsetA + i);
            v128_t vecB = wasm_v128_load(data + offsetB + i);
            v128_t product = wasm_f32x4_mul(vecA, vecB);
            sum = wasm_f32x4_add(sum, product);
        }

        float result[4];
        wasm_v128_store(result, sum);
        results[vectorIndex] = result[0] + result[1] + result[2] + result[3];
    }
}