import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';

const fileToDataUrl = async (filePath: string): Promise<string> => {
  try {
    const fileBuffer = await readFile(filePath);
    const base64Data = fileBuffer.toString('base64');
    return `data:application/octet-stream;base64,${base64Data}`;
  } catch (error) {
    console.error('Error reading file:', error);
    throw error;
  }
};

if (!existsSync('src/.gen')) {

  // create src/.gen folder
  mkdirSync('src/.gen', { recursive: true });
}

// Command to compile the C code with emscripten
const result = spawnSync('emcc', [
  'src/dot_product_simd.c',
  '-s',
  `EXPORTED_FUNCTIONS=["_dotProduct", "_malloc", "_free"]`,
  '-msimd128',
  '-O2',
  '-o',
  'src/.gen/dot_product.mjs',
  '--minify',
  '0',
  '-gsource-map',
  '--emit-tsd',
  'dot_product.d.ts'
], {
  stdio: 'inherit'
});

if (result.error) {
  console.error('Error executing emcc:', result.error);
} else {
  console.log('emcc command executed successfully');
}

const dataUrl = await fileToDataUrl('src/.gen/dot_product.wasm');

let dotProductRuntime = readFileSync('src/.gen/dot_product.mjs', 'utf-8')

// code injection and auto-deserialization of the wasm binary data
dotProductRuntime = dotProductRuntime.replace('var wasmBinaryFile;', `
const dataUrlToUint8Array = (dataUrl) => {
  // extract the base64 encoded part from the data URL
  const base64String = dataUrl.split(',')[1];
  
  // decode the base64 string into a binary string
  const binaryString = atob(base64String);

  // create a Uint8Array from the binary string
  const binaryLength = binaryString.length;
  const bytes = new Uint8Array(binaryLength);
  
  for (let i = 0; i < binaryLength; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};
var wasmBinaryFile = '${dataUrl}';
wasmBinary = dataUrlToUint8Array(wasmBinaryFile);
`)

writeFileSync('src/.gen/dot_product.mjs', dotProductRuntime);

const resultPkgRoll = spawnSync('pkgroll', {
  stdio: 'inherit'
});

if (resultPkgRoll.error) {
  console.error('Error executing emcc:', resultPkgRoll.error);
} else {
  console.log('pkgroll command executed successfully');
}
