// future work: https://github.com/webmachinelearning/webnn/blob/main/explainer.md

// we can (mis-)use general linear algebra function GEMM to speed-up matrice multiplications
const context = new MLContext();
const builder = new MLGraphBuilder(context);

// Create operands from the vectors
const tensorShape = [1, 1024];  // Shape for dot product: 1 row, 1024 columns
const a = builder.constant(tensorShape, vectorA);
const b = builder.constant(tensorShape, vectorB);

// Options for the GEMM operation
const options = {
  aTranspose: false,  // No need to transpose A
  bTranspose: true,   // Transpose B to match the dimensions for dot product
  alpha: 1.0,         // Scalar multiplier for the product
  beta: 0.0,          // Scalar multiplier for matrix C, not used here
  c: null             // No third matrix C
};

// Perform the dot product using gemm
const product = builder.gemm(a, b, options);

// Since the output is a tensor, you may need to fetch the result explicitly
const output = await context.compute(product);
console.log('Dot product:', output.data[0]);  // Output should be a single value tensor