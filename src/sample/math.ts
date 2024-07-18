import { vectorAData, vectorBData } from "./precomputed_samples";

// a simple PRNG based on a seed
export const seededRandom = (seed: number) => () => {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
};

export const seededShuffle = (array: Array<number>, seed: number) => {
  const random = seededRandom(seed);
  let currentIndex = array.length;
  const shuffledArray = array.slice();

  while (currentIndex !== 0) {
    const randomIndex = Math.floor(random() * currentIndex);
    currentIndex--;

    [shuffledArray[currentIndex], shuffledArray[randomIndex]] = [
      shuffledArray[randomIndex],
      shuffledArray[currentIndex],
    ];
  }
  return shuffledArray;
};

/* generate 2 x 20000 unique vectors á 1024 dimensions, seeded random */
export const generateSampleData = (
  seed = 31337,
  dims = 1024,
  size = 20000,
) => ({
  // generate 2 x 20000 unique vectors á 1024 dimensions, seeded
  vectorsA: Array.from({ length: size }, (_v: number, k) =>
    Float32Array.from(seededShuffle(vectorAData, seed + k)),
  ).map((d) => d.slice(0, dims)),
  vectorsB: Array.from({ length: size }, (_v: number, k) =>
    Float32Array.from(seededShuffle(vectorBData, seed + k)),
  ).map((d) => d.slice(0, dims)),
  dims,
});
