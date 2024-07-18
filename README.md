<span align="center">

  # fast-dotproduct

  ### Fast dot product calculations for the web platform.

</span>

> 🔬 My [experiments](./experiments/) have shown that the fastest algorithms to calculate the dot product between n-dimensional vectors currently
> are WebAssembly (SIMD) optimized and JIT-optimized JavaScript (unrolled loops for 16 byte aligned instructions to be vectorized by the optimizer) for workloads < 1 million vectors (aka. "a typical local vector store").

## ⚡ It's fast!

### WebAssembly, SIMD-optimized:
- Runs: `20000` single dot product calculations / pairs of n-dimensional vectors
- Took:
  - *1 ms* for `4` dimensions _(suffering from invocation overhead)_, 
  - *5 ms* for `384` dimensions, 
  - *7 ms* for `1024` dimensions

### JavaScript, JIT-optimized:
- Runs: `20000` single dot product calculations / pairs of n-dimensional vectors
- Took:
  - *0 ms* for `4` dimensions, 
  - *6 ms* for `384` dimensions, 
  - *16 ms* for `1024` dimensions

_Do you think, that you can improve these algos? Please help improving this project!_

## 📚 Usage

### 1. Install `fast-dotproduct`:

`npm/yarn/bun install fast-dotproduct`

### 2. Use it

```ts
import { dotProduct } from "fast-dotproduct"

const vectorA = new Float32Array([
  0.1785489022731781,
  0.6047865748405457,
  -0.29714474081993103,
  0.8343878388404846
])

const vectorB = new Float32Array([
  -0.12137561291456223,
  0.4885213375091553,
  0.3105606138706207,
  -0.23960202932357788
])

// -0.01842280849814415
const result = dotProduct([vectorA], [vectorB])

// if you have many vectors, pass them as a tensor for improved performance:
// const vectorsA = [Float32Array(1024), Float32Array(1024), ...]
// const vectorsB = [Float32Array(1024), Float32Array(1024), ...]
// const results = dotProduct(vectorsA, vectorsB)
// results[0] -> -0.01842280849814415
// results[...] -> etc.
```

Examples on how to use the WASM or JS-implementation specifically, please refer to [the tests](./src/index.spec.ts).

### 4. But wouldn't it be faster when we use the GPU?? 

Unfortunately, not necessarily. The time it takes to finish a program execution not only depends on the computation speed.
There are factors like `memory allocation` and `memory alignment` overhead. And when you're done with memory management,
there is still `shader compilation`. After you're done with the computation, you need to unpack/read the results back.
Using the GPU can be a great boost for heavy workloads, but isn't always benefitial for small ticket size workloads.


### 5. Can't we use `pthread` in WebAssembly and many Workers for parallel execution?

Well, you can, but it isn't necessarily faster either. Raming up a Worker can be pre-computed, but you still have to use the 
message loop to pass data from A to B and back, map memory and organize workloads. I couldn't find a scenario under the given
workfload ticket sizes, where `pthread` and Worker-based calculation wouldn't show a drastically bad impact on performance, unfortunately.

### 6. Help improve this project!

#### Setup

Clone this repo, install the dependencies (`bun` is recommended for speed), install [emscripten](https://emscripten.org/index.html) 
and run `npm run test` to verify the installation was successful. You may want to play with the experiments.


### Optimize/introduce new algos

Because of the current limits with the `WebGL`, `WebGL2` and `WebGPU` API's, including the `FP16` extension and `dot4U8packed` feature,
I couldn't find any implementation yet, that would be faster than simply passing a pointer of the JavaScript heap memory 1:1 
to the WebAssebly module that would use the Intel, AMD and ARM vector instruction sets to do the computation, passing back
the number. Not even passing chunks of memory or using consecutive or interleaved flat memory layouts yielded better results.
JIT-optimized vector calculation led to the V8 and JavaScriptCore optimizers to pick on the aligned memory/instructions and 
seemingly, to vectorize them automatically. The WebAssembly implementation did not benefit from `pthread` and the use of 
`Worker` in parallel.

Can you find any faster algorithm? Please share it via a Pull Request!
