// using SIMD and threads as both have > 90% coverage, even in browsers (browsers should use WebGL)
// usually, this implementation should be chosen for when running in a JS server-side environment,
// such as V8 in Node.js https://v8.dev/features/simd

// Wait for the WASM module to be instantiated
const Module = await new Promise((resolve) => {
    Module = {
        onRuntimeInitialized: () => resolve(Module)
    };
});

// Get the exported dotProduct function
const dotProduct = Module.cwrap('dotProduct', 'number', ['number', 'number', 'number']);

async function runWasm() {

    const vectorA = new Float32Array(1024).fill(1); // Example vector filled with 1s
    const vectorB = new Float32Array(1024).fill(1); // Example vector filled with 1s
    const length = vectorA.length;

    // Allocate memory in the WASM heap and copy the input vectors
    const ptrA = Module._malloc(vectorA.length * vectorA.BYTES_PER_ELEMENT);
    const ptrB = Module._malloc(vectorB.length * vectorB.BYTES_PER_ELEMENT);

    Module.HEAPF32.set(vectorA, ptrA / vectorA.BYTES_PER_ELEMENT);
    Module.HEAPF32.set(vectorB, ptrB / vectorB.BYTES_PER_ELEMENT);

    // Call the dotProduct function
    const result = dotProduct(ptrA, ptrB, length);

    // Free the allocated memory
    Module._free(ptrA);
    Module._free(ptrB);

    console.log('Dot Product:', result);
  }

  runWasm();