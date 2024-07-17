async function calculatePackedDotProductFP32(e1, e2) {
    // Dynamically create a hidden canvas
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    canvas.style.display = 'none';
    document.body.appendChild(canvas);
  
    const gl = canvas.getContext('webgl2');
    if (!gl) {
      throw new Error('WebGL2 is not supported in this browser.');
    }
  
    // Check if the floating point texture extension is available
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('FLOAT color buffer extension not available');
    }
  
    // Vertex shader source
    const vertexShaderSource = `
      attribute vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `;
  
    // Fragment shader source using GLSL
    const fragmentShaderSource = `
      precision highp float;
      uniform vec2 u_values;
      void main() {
        float dotProduct = dot(u_values, u_values);
        gl_FragColor = vec4(dotProduct, 0.0, 0.0, 1.0);
      }
    `;
  
    // Compile shaders
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  
    // Link the program
    const program = createProgram(gl, vertexShader, fragmentShader);
  
    // Set up a position buffer
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,
      1, -1,
      -1, 1,
      -1, 1,
      1, -1,
      1, 1,
    ]), gl.STATIC_DRAW);
  
    // Look up where the vertex data needs to go
    const positionLocation = gl.getAttribLocation(program, 'a_position');
  
    // Create a framebuffer to read the result
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  
    // Create a texture to store the result
    const resultTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, resultTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  
    // Attach the texture to the framebuffer
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, resultTexture, 0);
  
    // Check if the framebuffer is complete
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('Framebuffer is not complete');
    }
  
    // Set up the program
    gl.useProgram(program);
    gl.enableVertexAttribArray(positionLocation);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
  
    // Set the uniform values
    const valuesLocation = gl.getUniformLocation(program, 'u_values');
    gl.uniform2f(valuesLocation, e1, e2);
  
    // Draw the rectangle
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  
    // Read the result
    const pixels = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixels);
  
    // Clean up
    document.body.removeChild(canvas);
  
    // Return the dot product
    return pixels[0];
  }
  
  function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      throw new Error('Shader compilation failed');
    }
    return shader;
  }
  
  function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      throw new Error('Program linking failed');
    }
    return program;
  }
  
  // Example usage:
  calculatePackedDotProductFP32(0.123, 0.456)
    .then(result => console.log('Dot Product:', result))
    .catch(console.error);