// a simple PRNG based on a seed
export const seededRandom = (seed) => (() => {
    const x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
})
  
export const seededShuffle = (array, seed) => {
    const random = seededRandom(seed);
    let currentIndex = array.length;
    const shuffledArray = array.slice();

    while (currentIndex !== 0) {
        const randomIndex = Math.floor(random() * currentIndex);
        currentIndex--;

        [shuffledArray[currentIndex], shuffledArray[randomIndex]] = [
            shuffledArray[randomIndex], shuffledArray[currentIndex]
        ];
    }
    return shuffledArray;
}