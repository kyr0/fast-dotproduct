// n-dimensional vector
export type Vector = Float32Array;
export type Vectors = Array<Float32Array>;

export interface WasmModule {
  _malloc: (bytes: number) => number;
  _free: (ptr: number) => void;
  _dotProduct: (ptrA: number, ptrB: number, dims: number) => number;
  HEAPF32: {
    set: (data: Float32Array, offset: number) => void;
  };
}

export const dotProductWasm = (
  vectorsA: Vectors,
  vectorsB: Vectors,
  Module: WasmModule,
): Float32Array => {
  const dims = vectorsA[0].length;
  const size = vectorsA.length;
  const results = new Float32Array(size);
  const vectorByteSize = dims * Float32Array.BYTES_PER_ELEMENT;

  const ptrA = Module._malloc(vectorByteSize);
  const ptrB = Module._malloc(vectorByteSize);

  for (let i = 0; i < size; i++) {
    Module.HEAPF32.set(vectorsA[i], ptrA / Float32Array.BYTES_PER_ELEMENT);
    Module.HEAPF32.set(vectorsB[i], ptrB / Float32Array.BYTES_PER_ELEMENT);
    results[i] = Module._dotProduct(ptrA, ptrB, dims);
  }
  Module._free(ptrA);
  Module._free(ptrB);
  return results;
};

/** calculates the dot product of two n-dimensional vectors using an unrolled 4-element stepping (16 byte stepping) */
export const dotProductJS = (
  vectorsA: Vectors,
  vectorsB: Vectors,
): Float32Array => {
  const dims = vectorsA[0].length;
  const size = vectorsA.length;
  const results = new Float32Array(size);

  for (let i = 0; i < size; i++) {
    let result = 0.0;
    const vectorA = vectorsA[i];
    const vectorB = vectorsB[i];

    // Unrolling the loop to improve performance
    // 300% faster than baseline/naive: vectorA.reduce((sum, ai, i) => sum + ai * vectorB[i], 0)
    let j = 0;
    const unrollFactor = 4;
    const length = Math.floor(dims / unrollFactor) * unrollFactor;

    for (; j < length; j += unrollFactor) {
      // JIT optimization
      result +=
        vectorA[j] * vectorB[j] +
        vectorA[j + 1] * vectorB[j + 1] +
        vectorA[j + 2] * vectorB[j + 2] +
        vectorA[j + 3] * vectorB[j + 3];
    }
    results[i] = result;
  }
  return results;
};

// 380% to 300% faster than baseline JS
export const dotProduct = (
  vectorsA: Vectors,
  vectorsB: Vectors,
  Module?: any,
): Float32Array => {
  if (vectorsA.length !== vectorsB.length) {
    throw new Error("Dimensionality of both vectors must be equal.");
  }
  if (vectorsA.length === 0) {
    throw new Error("Vectors are empty.");
  }
  if (vectorsA.length < 4) {
    throw new Error("The minimum dimensionality for optimizations is 4.");
  }
  const dims = vectorsA[0].length;

  if (dims % 4 > 0) {
    throw new Error("Dimensionality for must be a multiple of 4.");
  }
  return typeof WebAssembly === "object"
    ? dotProductWasm(vectorsA, vectorsB, Module)
    : dotProductJS(vectorsA, vectorsB);
};
