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