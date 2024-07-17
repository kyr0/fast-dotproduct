import { vectorAData, vectorBData } from "./sample-vectors.mjs"
import { seededShuffle } from "./math.mjs"

// generate 2 x 20000 unique vectors á 1024 dimensions, seeded
export const generateSampleData = (seed = 31337, dims = 1024) => ({
// generate 2 x 20000 unique vectors á 1024 dimensions, seeded
    vectorsA: Array.from({ length: 20000 }, (v, k) => Float32Array.from(seededShuffle(vectorAData, seed + k))).map(d => d.slice(0, dims)),
    vectorsB: Array.from({ length: 20000 }, (v, k) => Float32Array.from(seededShuffle(vectorBData, seed + k))).map(d => d.slice(0, dims)),
    dims: 1024
})