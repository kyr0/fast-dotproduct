<span align="center">

  # fast-dotproduct

  ### Fast dot product calculations for the web platform. Speeds up your 🐌 dot product calculations by up to 103457% ⚡.

</span>

> 🔬 My [experiments](./experiments/) have shown that the fastest algorithms to calculate the dot product between n-dimensional vectors currently
> are WebAssembly (SIMD) optimized and JIT-optimized JavaScript (unrolled loops for 16 byte aligned instructions to be vectorized by the optimizer) for workloads < 1 million vectors (aka. "a typical local vector store").

## ⚡ It's fast!

### 🐌 Slow: baseline/naive JS:
- Runs: `100k` single dot product calculations on 2 n-dimensional vectors, loop-inlined
- Took:
  - *35756 ms* for 4 dimensions, 
  - *36012 ms* ms for 384 dimensions, 
  - *36525 ms* ms for 1024 dimensions

### 🦆 Faster: pure JavaScript, but JIT-optimized: (468,27x faster, or 46827% faster)
- Runs: `100k` single dot product calculations on 2 n-dimensional vectors, loop-inlined
- Took:
  - *1 ms* for `4` dimensions,
  - *30 ms* for `384` dimensions, 
  - *78 ms* for `1024` dimensions

### 🐇 Fastest: WebAssembly, SIMD-optimized (1034,57x faster, or 103457%):
- Runs: `100k` single dot product calculations on 2 n-dimensional vectors, loop-inlined
- Took:
  - *3 ms* for `4` dimensions _(suffering from inital invocation overhead)_, 
  - *13 ms* for `384` dimensions, 
  - *35 ms* for `1024` dimensions

_Do you see any potential for further improvements? Please contribute to this project! Let's build the fastest dotproduct library for the web!_

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
const result = await dotProduct([vectorA], [vectorB])

// if you have many vectors, pass them as a tensor for improved performance:
// const vectorsA = [Float32Array(1024), Float32Array(1024), ...]
// const vectorsB = [Float32Array(1024), Float32Array(1024), ...]
// const results = dotProduct(vectorsA, vectorsB)
// results[0] -> -0.01842280849814415
// results[...] -> etc.
```

Examples on how to use the WASM or JS-implementation specifically, please refer to [the tests](./src/index.spec.ts).

### Why shouldn't I use a simple, naive dot product implementation in JS?

> "Wouldn't a simple implementation not do it? I read that the V8 and JavaScriptCore optimizers can do a great job, optimizing!"

```ts
vectorA.reduce((sum, ai, j) => sum + ai * vectorB[j], 0)
```

Well.. unfortunately, sometimes they do, sometimes they don't. If we narrow the scope of what the optimizer need to speculate about,
we get a performance boost. This is, what happens in this repo's JIT optimized implementation. But it's still a JS/JIT based implementation.
Using a WebAssembly runtime, and explicitly using the `v128` instruction set, as well as explicitly choosing the instructions to use, 
and on top of that, calculating 4 float32 dot product calculations per instruction, we get the greatest boost.

You can run the performance tests on your machine. Checkout the repo and run: `npm run test`.

### Err, cool, but for what do I need this anyway?

Vector search, for example. [Dot products](https://en.wikipedia.org/wiki/Dot_product) are an essential part of the math behind [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

### But wouldn't it be faster when we use the GPU?? 

Unfortunately, not necessarily. The time it takes to finish a program execution not only depends on the computation speed.
There are factors like `memory allocation` and `memory alignment` overhead. And when you're done with memory management,
there is still `shader compilation`. After you're done with the computation, you need to unpack/read the results back.
Using the GPU can be a great boost for heavy workloads, but isn't always benefitial for small ticket size workloads.


### Can't we use `pthread` in WebAssembly and many Workers for parallel execution?

Well, you can, but it isn't necessarily faster either. Raming up a Worker can be pre-computed, but you still have to use the 
message loop to pass data from A to B and back, map memory and organize workloads. I couldn't find a scenario under the given
workfload ticket sizes, where `pthread` and Worker-based calculation wouldn't show a drastically bad impact on performance, unfortunately.

### Help improve this project!

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
