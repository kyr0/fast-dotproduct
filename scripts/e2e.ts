import { dotProductWasm, initWasm } from "../dist/index.mjs";

// @ts-ignore
import getWasmModule from "../src/.gen/dot_product.mjs";

// make sure the Module is loaded
await initWasm();

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
const result = dotProductWasm([vectorA], [vectorB])

console.log("result", result)

// make sure the Module is loaded with custom loaded module
await initWasm(await getWasmModule());


const vectorA1 = new Float32Array([
  0.1785489022731781,
  0.6047865748405457,
  -0.29714474081993103,
  0.8343878388404846
])

const vectorB1 = new Float32Array([
  -0.12137561291456223,
  0.4885213375091553,
  0.3105606138706207,
  -0.23960202932357788
])

// -0.01842280849814415
const result1 = dotProductWasm([vectorA1], [vectorB1])

console.log("result1", result1)