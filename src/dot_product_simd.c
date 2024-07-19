#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

float dotProduct(const float *a, const float *b, size_t dims) {
    v128_t sum = wasm_f32x4_splat(0.0f);

    // process in chunks of 4, memory/instruction alignment
    for (size_t i = 0; i < dims; i += 4) {
        sum = wasm_f32x4_add(sum, 
            wasm_f32x4_mul(
                wasm_v128_load(&a[i]), 
                wasm_v128_load(&b[i])
            )
        );
    }

    // extract from the SIMD register and sum up directly
    return wasm_f32x4_extract_lane(sum, 0) + 
           wasm_f32x4_extract_lane(sum, 1) + 
           wasm_f32x4_extract_lane(sum, 2) + 
           wasm_f32x4_extract_lane(sum, 3); 
}