// https://developer.chrome.com/blog/new-in-webgpu-123#dp4a_built-in_functions_support_in_wgsl
// https://www.w3.org/TR/WGSL/#dot4U8Packed-builtin
async function calculatePackedDotProduct(e1, e2) {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser.');
  }

  if (!navigator.gpu.wgslLanguageFeatures.has("packed_4x8_integer_dot_product")) {
    throw new Error('DP4a built-in functions are not available');
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

  // Shader code using dot4U8Packed
  const shaderCode = `
    requires packed_4x8_integer_dot_product;

    @group(0) @binding(0) var<storage, read_write> result: array<u32>;

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let e1: u32 = ${e1}u;
      let e2: u32 = ${e2}u;
      result[0] = dot4U8Packed(e1, e2);
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
calculatePackedDotProduct(16909060, 33752069) 
  .then(result => console.log('Dot Product:', result))
  .catch(console.error);