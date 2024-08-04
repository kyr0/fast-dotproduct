import getWasmModule from "./.gen/dot_product";

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

let Module: WasmModule;

export const initWasm = async (module?: WasmModule): Promise<WasmModule> => {
  if (module) {
    Module = module;
  }

  if (!Module) {
    Module = await getWasmModule();
  }
  return Module;
};

export const singleDotProductWasm = (
  vectorA: Vector,
  vectorB: Vector,
): number => {
  const dims = vectorA.length;
  const results = new Float32Array(1);
  const vectorByteSize = dims * Float32Array.BYTES_PER_ELEMENT;

  const ptrA = Module._malloc(vectorByteSize);
  const ptrB = Module._malloc(vectorByteSize);

  Module.HEAPF32.set(vectorA, ptrA / Float32Array.BYTES_PER_ELEMENT);
  Module.HEAPF32.set(vectorB, ptrB / Float32Array.BYTES_PER_ELEMENT);
  results[0] = Module._dotProduct(ptrA, ptrB, dims);

  Module._free(ptrA);
  Module._free(ptrB);

  return results[0];
};

export const dotProductWasm = (
  vectorsA: Vectors,
  vectorsB: Vectors,
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

export const singleDotProductJS = (
  vectorA: Vector,
  vectorB: Vector,
): number => {
  const dims = vectorA.length;

  let result = 0.0;

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
  return result;
};

export const dotProductNaiveBaselineJS = (
  vectorsA: Vectors,
  vectorsB: Vectors,
): Float32Array => {
  const size = vectorsA.length;
  const results = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    results[i] = vectorsA[i].reduce(
      (sum, ai, j) => sum + ai * vectorsB[i][j],
      0,
    );
  }
  return results;
};

// 380% to 300% faster than baseline JS
export const dotProduct = (
  vectorsA: Vectors,
  vectorsB: Vectors,
): Float32Array => {
  if (vectorsA.length === 0 || vectorsB.length === 0) {
    throw new Error("One of the vector arrays are empty.");
  }
  const dimsA0 = vectorsA[0].length;
  const dimsB0 = vectorsB[0].length;

  if (dimsA0 !== dimsB0) {
    throw new Error(
      "Dimensionality of both vectors must be equal (only tested the first one).",
    );
  }

  if (dimsA0 < 4) {
    throw new Error("The minimum dimensionality for optimizations is 4.");
  }

  if (dimsA0 % 4 !== 0) {
    throw new Error("Dimensionality for must be a multiple of 4.");
  }

  // TODO: probably should test for SIMD support too, although it's > 94% evergreen already
  return typeof WebAssembly === "object" && Module
    ? dotProductWasm(vectorsA, vectorsB)
    : dotProductJS(vectorsA, vectorsB);
};

// 380% to 300% faster than baseline JS
export const singleDotProduct = (vectorA: Vector, vectorB: Vector): number => {
  if (vectorA.length === 0 || vectorB.length === 0) {
    throw new Error("One of the vector arrays are empty.");
  }
  const dimsA = vectorA.length;
  const dimsB = vectorB.length;

  if (dimsA !== dimsB) {
    throw new Error("Dimensionality of both vectors must be equal.");
  }

  if (dimsA < 4) {
    throw new Error("The minimum dimensionality for optimizations is 4.");
  }

  if (dimsA % 4 !== 0) {
    throw new Error("Dimensionality for must be a multiple of 4.");
  }

  // TODO: probably should test for SIMD support too, although it's > 94% evergreen already
  return typeof WebAssembly === "object" && Module
    ? singleDotProductWasm(vectorA, vectorB)
    : singleDotProductJS(vectorA, vectorB);
};
