// standard WebGPU, f32
// https://www.w3.org/TR/WGSL/#dot-builtin
async function calculatePackedDotProductFP32(e1, e2) {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser.');
    }
  
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
  
    // Create buffer for result storage
    const resultStorageBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  
    // Create buffer for result copy
    const resultCopyBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const shaderCode = `
        @group(0) @binding(0) var<storage, read_write> result: array<f32, 1>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let e1: vec2<f32> = vec2<f32>(f32(${e1}), f32(${e2}));
        let e2: vec2<f32> = vec2<f32>(f32(${e1}), f32(${e2}));
        result[0] = dot(e1, e2);
        }
    `;
  
    const shaderModule = device.createShaderModule({ code: shaderCode });
  
    // Compute pipeline
    const computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
  
    // Bind group
    const bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: resultStorageBuffer } }
      ]
    });
  
    // Command encoding and dispatch
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();
  
    // Copy the result to the resultCopyBuffer
    commandEncoder.copyBufferToBuffer(resultStorageBuffer, 0, resultCopyBuffer, 0, Uint32Array.BYTES_PER_ELEMENT);
  
    // Submit commands
    device.queue.submit([commandEncoder.finish()]);
  
    // Read back the results
    await resultCopyBuffer.mapAsync(GPUMapMode.READ);
    const resultsArray = new Uint32Array(resultCopyBuffer.getMappedRange());
    const result = resultsArray[0];
    resultCopyBuffer.unmap();
  
    return result;
  }
  
  // Example usage:
  calculatePackedDotProductFP32(0.123, 0.456)
    .then(result => console.log('Dot Product:', result))
    .catch(console.error);





    /**
     * 
     * Working candidate (almost)
     * 
     * 
     * if (!navigator.gpu) {
  throw new Error('WebGPU is not supported in this browser.');
}

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Helper function to create a buffer
function createBuffer(device, data, usage) {
  const array = new Float32Array(data);
  const paddedSize = Math.ceil(array.byteLength / 16) * 16; // Ensure the buffer size is a multiple of 16 bytes
  const buffer = device.createBuffer({
    size: paddedSize,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(array);
  buffer.unmap();
  return buffer;
}

// Function to flatten and prepare data correctly
function prepareData(dataArrays) {
  return new Float32Array(dataArrays.reduce((acc, val) => acc.concat(Array.from(val)), []));
}

// Function to pad data to ensure correct alignment and meet minimum buffer size
function padData(data, alignment) {
  const byteCount = data.length * Float32Array.BYTES_PER_ELEMENT;
  const paddedByteCount = Math.max(Math.ceil(byteCount / alignment) * alignment, 32); // Ensure at least 32 bytes to meet min size requirements
  const floatCount = paddedByteCount / Float32Array.BYTES_PER_ELEMENT;
  const paddedData = new Float32Array(floatCount);
  paddedData.set(data);
  return paddedData;
}

// Example input data
const e1 = [
  new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
  new Float32Array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
];
const e2 = [
  new Float32Array([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]),
  new Float32Array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
];

const flatE1 = prepareData(e1);
const flatE2 = prepareData(e2);

const paddedE1 = padData(flatE1, 16); // Ensuring each segment starts on a 16-byte boundary
const paddedE2 = padData(flatE2, 16);

const e1Buffer = createBuffer(device, paddedE1, GPUBufferUsage.STORAGE);
const e2Buffer = createBuffer(device, paddedE2, GPUBufferUsage.STORAGE);

// Create buffer for result storage, padding to ensure alignment
const resultStorageBuffer = device.createBuffer({
  size: Math.max(16 * e1.length, 32), // Each float result is padded to 16 bytes, ensuring minimum size
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// Create buffer for result copy, padding to ensure alignment
const resultCopyBuffer = device.createBuffer({
  size: Math.max(16 * e1.length, 32), // Each float result is padded to 16 bytes, ensuring minimum size
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const shaderCode = `
  @group(0) @binding(0) var<storage, read> e1: array<vec4<f32>, 2>;
  @group(0) @binding(1) var<storage, read> e2: array<vec4<f32>, 2>;
  @group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>, 1>;

  @compute @workgroup_size(1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pairIndex = global_id.x;
    var dot_product: f32 = 0.0;
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
      dot_product += dot(e1[pairIndex * 2u + i], e2[pairIndex * 2u + i]);
    }
    result[pairIndex] = vec4<f32>(dot_product, 0.0, 0.0, 0.0);
  }
`;

const shaderModule = device.createShaderModule({ code: shaderCode });

// Compute pipeline
const computePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: shaderModule,
    entryPoint: 'main'
  }
});

// Bind group
const bindGroup = device.createBindGroup({
  layout: computePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: e1Buffer } },
    { binding: 1, resource: { buffer: e2Buffer } },
    { binding: 2, resource: { buffer: resultStorageBuffer } }
  ]
});

// Command encoding and dispatch
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(e1.length);
passEncoder.end();

// Copy the result to the resultCopyBuffer
commandEncoder.copyBufferToBuffer(resultStorageBuffer, 0, resultCopyBuffer, 0, resultStorageBuffer.size);

// Submit commands
device.queue.submit([commandEncoder.finish()]);

// Read back the results
await resultCopyBuffer.mapAsync(GPUMapMode.READ);
const resultsArray = new Float32Array(resultCopyBuffer.getMappedRange());
const finalResults = new Float32Array(e1.length);
for (let i = 0; i < e1.length; i++) {
  finalResults[i] = resultsArray[i * 4]; // Extract the x component
}
resultCopyBuffer.unmap();

console.log('Final Results:', finalResults);
     */


/**
 * second candidate
 * 
 * if (!navigator.gpu) {
  throw new Error('WebGPU is not supported in this browser.');
}

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Helper function to create a buffer
function createBuffer(device, data, usage) {
  const array = new Float32Array(data);
  const paddedSize = Math.ceil(array.byteLength / 16) * 16; // Ensure the buffer size is a multiple of 16 bytes
  const buffer = device.createBuffer({
    size: paddedSize,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(array);
  buffer.unmap();
  return buffer;
}

// Function to flatten and prepare data correctly
function prepareData(dataArrays) {
  return new Float32Array(dataArrays.reduce((acc, val) => acc.concat(Array.from(val)), []));
}

// Function to pad data to ensure correct alignment and meet minimum buffer size
function padData(data, alignment) {
  const byteCount = data.length * Float32Array.BYTES_PER_ELEMENT;
  const paddedByteCount = Math.max(Math.ceil(byteCount / alignment) * alignment, 32); // Ensure at least 32 bytes to meet min size requirements
  const floatCount = paddedByteCount / Float32Array.BYTES_PER_ELEMENT;
  const paddedData = new Float32Array(floatCount);
  paddedData.set(data);
  return paddedData;
}

// Example input data - replace these arrays with your actual data
const e1 = [
  new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
  new Float32Array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
  new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
  // Add more vector pairs as needed
];
const e2 = [
  new Float32Array([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]),
  new Float32Array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
  new Float32Array([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]),
  // Add more vector pairs as needed
];

const flatE1 = prepareData(e1);
const flatE2 = prepareData(e2);

const paddedE1 = padData(flatE1, 16); // Ensuring each segment starts on a 16-byte boundary
const paddedE2 = padData(flatE2, 16);

const e1Buffer = createBuffer(device, paddedE1, GPUBufferUsage.STORAGE);
const e2Buffer = createBuffer(device, paddedE2, GPUBufferUsage.STORAGE);

const numVectors = e1.length; // Number of vector pairs

// Create buffer for result storage, padding to ensure alignment
const resultStorageBuffer = device.createBuffer({
  size: Math.max(16 * numVectors, 32), // Each float result is padded to 16 bytes, ensuring minimum size
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// Create buffer for result copy, padding to ensure alignment
const resultCopyBuffer = device.createBuffer({
  size: Math.max(16 * numVectors, 32), // Each float result is padded to 16 bytes, ensuring minimum size
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const shaderCode = `
  @group(0) @binding(0) var<storage, read> e1: array<vec4<f32>, ${2 * numVectors}>;
  @group(0) @binding(1) var<storage, read> e2: array<vec4<f32>, ${2 * numVectors}>;
  @group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>, ${numVectors}>;

  @compute @workgroup_size(1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pairIndex = global_id.x;
    var dot_product: f32 = 0.0;
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
      dot_product += dot(e1[pairIndex * 2u + i], e2[pairIndex * 2u + i]);
    }
    result[pairIndex] = vec4<f32>(dot_product, 0.0, 0.0, 0.0);
  }
`;

const shaderModule = device.createShaderModule({ code: shaderCode });

// Compute pipeline
const computePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: shaderModule,
    entryPoint: 'main'
  }
});

// Bind group
const bindGroup = device.createBindGroup({
  layout: computePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: e1Buffer } },
    { binding: 1, resource: { buffer: e2Buffer } },
    { binding: 2, resource: { buffer: resultStorageBuffer } }
  ]
});

// Command encoding and dispatch
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(numVectors);
passEncoder.end();

// Copy the result to the resultCopyBuffer
commandEncoder.copyBufferToBuffer(resultStorageBuffer, 0, resultCopyBuffer, 0, resultStorageBuffer.size);

// Submit commands
device.queue.submit([commandEncoder.finish()]);

// Read back the results
await resultCopyBuffer.mapAsync(GPUMapMode.READ);
const resultsArray = new Float32Array(resultCopyBuffer.getMappedRange());
const finalResults = new Float32Array(numVectors);
for (let i = 0; i < numVectors; i++) {
  finalResults[i] = resultsArray[i * 4]; // Extract the x component
}
resultCopyBuffer.unmap();

console.log('Final Results:', finalResults);
 * 
 * 
 */



/*** READY:
 
if (!navigator.gpu) {
  throw new Error('WebGPU is not supported in this browser.');
}

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Helper function to create a buffer
function createBuffer(device, data, usage) {
  const array = new Float32Array(data);
  const paddedSize = Math.ceil(array.byteLength / 16) * 16; // Ensure the buffer size is a multiple of 16 bytes
  const buffer = device.createBuffer({
    size: paddedSize,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(array);
  buffer.unmap();
  return buffer;
}

// Function to flatten and prepare data correctly
function prepareData(dataArrays) {
  return new Float32Array(dataArrays.reduce((acc, val) => acc.concat(Array.from(val)), []));
}

// Function to pad data to ensure correct alignment and meet minimum buffer size
function padData(data, alignment) {
  const byteCount = data.length * Float32Array.BYTES_PER_ELEMENT;
  const paddedByteCount = Math.max(Math.ceil(byteCount / alignment) * alignment, 32); // Ensure at least 32 bytes to meet min size requirements
  const floatCount = paddedByteCount / Float32Array.BYTES_PER_ELEMENT;
  const paddedData = new Float32Array(floatCount);
  paddedData.set(data);
  return paddedData;
}

// Example input data - replace these arrays with your actual data
const e1 = [
  new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]),
  new Float32Array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 5.0, 6.0, 7.0, 8.0]),
  new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]),
  // Add more vector pairs as needed
];
const e2 = [
  new Float32Array([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
  new Float32Array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 5.0, 6.0, 7.0, 8.0]),
  new Float32Array([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
  // Add more vector pairs as needed
];

const flatE1 = prepareData(e1);
const flatE2 = prepareData(e2);

const paddedE1 = padData(flatE1, 16); // Ensuring each segment starts on a 16-byte boundary
const paddedE2 = padData(flatE2, 16);

const e1Buffer = createBuffer(device, paddedE1, GPUBufferUsage.STORAGE);
const e2Buffer = createBuffer(device, paddedE2, GPUBufferUsage.STORAGE);

const numVectors = e1.length; // Number of vector pairs

// Create buffer for result storage, padding to ensure alignment
const resultStorageBuffer = device.createBuffer({
  size: Math.max(16 * numVectors, 32), // Each float result is padded to 16 bytes, ensuring minimum size
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});

// Create buffer for result copy, padding to ensure alignment
const resultCopyBuffer = device.createBuffer({
  size: Math.max(16 * numVectors, 32), // Each float result is padded to 16 bytes, ensuring minimum size
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const shaderCode = `
  @group(0) @binding(0) var<storage, read> e1: array<vec4<f32>, ${2 * numVectors}>;
  @group(0) @binding(1) var<storage, read> e2: array<vec4<f32>, ${2 * numVectors}>;
  @group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>, ${numVectors}>;

  @compute @workgroup_size(1)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pairIndex = global_id.x;
    var dot_product: f32 = 0.0;
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
      dot_product += dot(e1[pairIndex * 2u + i], e2[pairIndex * 2u + i]);
    }
    result[pairIndex] = vec4<f32>(dot_product, 0.0, 0.0, 0.0);
  }
`;

const shaderModule = device.createShaderModule({ code: shaderCode });

// Compute pipeline
const computePipeline = device.createComputePipeline({
  layout: "auto",
  compute: {
    module: shaderModule,
    entryPoint: 'main'
  }
});

// Bind group
const bindGroup = device.createBindGroup({
  layout: computePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: e1Buffer } },
    { binding: 1, resource: { buffer: e2Buffer } },
    { binding: 2, resource: { buffer: resultStorageBuffer } }
  ]
});

// Command encoding and dispatch
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(computePipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(numVectors);
passEncoder.end();

// Copy the result to the resultCopyBuffer
commandEncoder.copyBufferToBuffer(resultStorageBuffer, 0, resultCopyBuffer, 0, resultStorageBuffer.size);

// Submit commands
device.queue.submit([commandEncoder.finish()]);

// Read back the results
await resultCopyBuffer.mapAsync(GPUMapMode.READ);
const resultsArray = new Float32Array(resultCopyBuffer.getMappedRange());
const finalResults = new Float32Array(numVectors);
for (let i = 0; i < numVectors; i++) {
  finalResults[i] = resultsArray[i * 4]; // Extract the x component
}
resultCopyBuffer.unmap();

console.log('Final Results:', finalResults);

 */