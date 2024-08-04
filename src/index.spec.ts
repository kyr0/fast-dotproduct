import { expect, test } from "vitest";
import {
  dotProduct,
  dotProductJS,
  dotProductNaiveBaselineJS,
  dotProductWasm,
  initWasm,
  singleDotProduct,
  singleDotProductJS,
  singleDotProductWasm,
} from "./";
import { generateSampleData } from "./sample/math";
import { perf } from "@jsheaven/perf";

// @ts-ignore
import getWasmModule from "./.gen/dot_product.mjs";

// make sure the Module is loaded
await initWasm(await getWasmModule());

// 2 x 1024 float32 vectors with 1024 dimensions, seeded random
const sampleData20kx1024dims = generateSampleData(
  31337 /* seed */,
  1024 /* dimensions */,
  100000 /* samples */,
);

const sampleData20kx384dims = generateSampleData(
  31337 /* seed */,
  384 /* dimensions */,
  100000 /* samples */,
);

const sampleData20kx4dims = generateSampleData(
  31337 /* seed */,
  4 /* dimensions */,
  100000 /* samples */,
);

const samplesPerDimension = {
  4: sampleData20kx4dims,
  384: sampleData20kx384dims,
  1024: sampleData20kx1024dims,
};

test("Make sure the API interface/contract is fulfilled", async () => {
  expect(typeof initWasm).toEqual("function");
  expect(typeof dotProduct).toEqual("function");
  expect(typeof dotProductJS).toEqual("function");
  expect(typeof dotProductWasm).toEqual("function");
  expect(typeof dotProductNaiveBaselineJS).toEqual("function");
  expect(typeof singleDotProduct).toEqual("function");
  expect(typeof singleDotProductJS).toEqual("function");
  expect(typeof singleDotProductWasm).toEqual("function");
});

test("Calculates the dot product of two vectors using naive/baseline JS", async () => {
  const vectorA = sampleData20kx4dims.vectorsA[0];
  console.log("vectorA", vectorA);
  const vectorB = sampleData20kx4dims.vectorsB[0];
  console.log("vectorB", vectorB);
  const results = dotProductNaiveBaselineJS([vectorA], [vectorB]);
  expect(results[0]).toBeCloseTo(-0.01842280849814415);
});

test("Calculates the dot product of two vectors using JIT optimized JS", async () => {
  const vectorA = sampleData20kx4dims.vectorsA[0];
  console.log("vectorA", vectorA);
  const vectorB = sampleData20kx4dims.vectorsB[0];
  console.log("vectorB", vectorB);
  const results = dotProductJS([vectorA], [vectorB]);
  expect(results[0]).toBeCloseTo(-0.01842280849814415);
});

test("Calculates the dot product of two vectors using SIMD optimized WebAssembly module", async () => {
  const vectorA = sampleData20kx4dims.vectorsA[0];
  const vectorB = sampleData20kx4dims.vectorsB[0];
  const results = dotProductWasm([vectorA], [vectorB]);
  expect(results[0]).toBeCloseTo(-0.01842280849814415);
});

test("Calculates the dot product of two vectors using single JIT optimized JS", async () => {
  const vectorA = sampleData20kx4dims.vectorsA[0];
  const vectorB = sampleData20kx4dims.vectorsB[0];
  const result = singleDotProductJS(vectorA, vectorB);
  expect(result).toBeCloseTo(-0.01842280849814415);
});

test("Calculates the dot product of two vectors using single SIMD optimized WebAssembly module", async () => {
  const vectorA = sampleData20kx4dims.vectorsA[0];
  const vectorB = sampleData20kx4dims.vectorsB[0];
  const result = singleDotProductWasm(vectorA, vectorB);
  expect(result).toBeCloseTo(-0.01842280849814415);
});

test("Calculates the dot product of two vectors using auto-select algo", async () => {
  const vectorA = sampleData20kx4dims.vectorsA[0];
  const vectorB = sampleData20kx4dims.vectorsB[0];
  const result = singleDotProduct(vectorA, vectorB);
  expect(result).toBeCloseTo(-0.01842280849814415);
});

test("Auto-switches between JIT-optimized JS and SIMD-optimized WASM based on WebAssembly availability", async () => {
  const vectorA = sampleData20kx4dims.vectorsA[0];
  const vectorB = sampleData20kx4dims.vectorsB[0];
  const results = dotProduct(
    [vectorA, vectorA, vectorA, vectorA],
    [vectorB, vectorB, vectorB, vectorB],
  );
  expect(results[0]).toBeCloseTo(-0.01842280849814415);
});

test("perf: Measure and report the performance of 100000 single, naive/baseline JS-based dot product calculations", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 100000;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "JS_JIT_single",
        fn: async (dims: number, i: number) => {
          const vectorA = sampleData20kx1024dims.vectorsA[i].slice(0, dims);
          const vectorB = sampleData20kx1024dims.vectorsB[i].slice(0, dims);
          if (!times[dims]) {
            times[dims] = performance.now();
          }
          dotProductNaiveBaselineJS([vectorA], [vectorB]);

          if (i === iterations - 1 && times[dims]) {
            times[dims] = performance.now() - times[dims];
          }
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## JavaScript, JIT-optimized:
- Runs: ${iterations} single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, JIT optimized JS-based dot product calculations", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 100000;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "JS_JIT_multi_single",
        fn: async (dims: number, i: number) => {
          const vectorA = sampleData20kx1024dims.vectorsA[i].slice(0, dims);
          const vectorB = sampleData20kx1024dims.vectorsB[i].slice(0, dims);
          if (!times[dims]) {
            times[dims] = performance.now();
          }
          dotProductJS([vectorA], [vectorB]);

          if (i === iterations - 1 && times[dims]) {
            times[dims] = performance.now() - times[dims];
          }
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## JavaScript, JIT-optimized:
- Runs: ${iterations} single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, SIMD-optimized WASM-based dot product calculations", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 100000;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "WASM_single",
        fn: async (dims: number, i: number) => {
          const vectorA = sampleData20kx1024dims.vectorsA[i].slice(0, dims);
          const vectorB = sampleData20kx1024dims.vectorsB[i].slice(0, dims);
          if (!times[dims]) {
            times[dims] = performance.now();
          }
          dotProductWasm([vectorA], [vectorB]);

          if (i === iterations - 1 && times[dims]) {
            times[dims] = performance.now() - times[dims];
          }
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## WebAssembly, SIMD-optimized:
- Runs: ${iterations} single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, naive/baseline JS-based dot product calculations, passed at once", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 1;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "JS_multi",
        fn: async (dims: number, _i: number) => {
          const vectorsA = samplesPerDimension[dims as 4 | 384 | 1024].vectorsA;
          const vectorsB = samplesPerDimension[dims as 4 | 384 | 1024].vectorsB;
          times[dims] = performance.now();
          dotProductNaiveBaselineJS(vectorsA, vectorsB);
          times[dims] = performance.now() - times[dims];
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## JavaScript, naive, baseline:
- Runs: 100000 single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, JIT-optimized JS-based dot product calculations, passed at once", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 1;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "JS_JIT_multi",
        fn: async (dims: number, _i: number) => {
          const vectorsA = samplesPerDimension[dims as 4 | 384 | 1024].vectorsA;
          const vectorsB = samplesPerDimension[dims as 4 | 384 | 1024].vectorsB;
          times[dims] = performance.now();
          dotProductJS(vectorsA, vectorsB);
          times[dims] = performance.now() - times[dims];
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## JavaScript, JIT-optimized:
- Runs: 100000 single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, SIMD-optimized WASM-based dot product calculations, passed at once", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 1;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "WASM_multi",
        fn: async (dims: number, _i: number) => {
          const vectorsA = samplesPerDimension[dims as 4 | 384 | 1024].vectorsA;
          const vectorsB = samplesPerDimension[dims as 4 | 384 | 1024].vectorsB;
          times[dims] = performance.now();
          dotProductWasm(vectorsA, vectorsB);
          times[dims] = performance.now() - times[dims];
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## WebAssembly, SIMD-optimized:
- Runs: 100000 single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, naive/baseline JS-based dot product calculations (atomic vector API)", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 100000;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "JS_naive_single",
        fn: async (dims: number, i: number) => {
          const vectorA = sampleData20kx1024dims.vectorsA[i].slice(0, dims);
          const vectorB = sampleData20kx1024dims.vectorsB[i].slice(0, dims);
          if (!times[dims]) {
            times[dims] = performance.now();
          }
          singleDotProduct(vectorA, vectorB);

          if (i === iterations - 1 && times[dims]) {
            times[dims] = performance.now() - times[dims];
          }
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## JavaScript, naive, baseline:
- Runs: ${iterations} single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, JIT optimized JS-based dot product calculations (atomic vector API)", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 100000;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "JS_JIT_single",
        fn: async (dims: number, i: number) => {
          const vectorA = sampleData20kx1024dims.vectorsA[i].slice(0, dims);
          const vectorB = sampleData20kx1024dims.vectorsB[i].slice(0, dims);
          if (!times[dims]) {
            times[dims] = performance.now();
          }
          singleDotProductJS(vectorA, vectorB);

          if (i === iterations - 1 && times[dims]) {
            times[dims] = performance.now() - times[dims];
          }
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## JavaScript, JIT-optimized:
- Runs: ${iterations} single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});

test("perf: Measure and report the performance of 100000 single, SIMD-optimized WASM-based dot product calculations (atomic vector API)", async () => {
  const times: { [index: number]: number } = {};
  const iterations = 100000;
  const dimensions = [4, 384, 1024];
  await perf(
    [
      {
        name: "WASM_single",
        fn: async (dims: number, i: number) => {
          const vectorA = sampleData20kx1024dims.vectorsA[i].slice(0, dims);
          const vectorB = sampleData20kx1024dims.vectorsB[i].slice(0, dims);
          if (!times[dims]) {
            times[dims] = performance.now();
          }
          singleDotProductWasm(vectorA, vectorB);

          if (i === iterations - 1 && times[dims]) {
            times[dims] = performance.now() - times[dims];
          }
        },
      },
    ],
    dimensions /* sizes (dimensionality) */,
    true /* warmup*/,
    iterations /* iterations */,
    30000 /* maxExecutionTime */,
    true /* auto-optimize chuck size */,
  );

  console.log(`# Results:
## WebAssembly, SIMD-optimized:
- Runs: ${iterations} single dot product calculations / pairs of n-dimensional vectors
- Took:
${dimensions.map((d) => `  - ${times[d].toFixed()} ms for ${d} dimensions`).join(", \n")}\n`);
});
