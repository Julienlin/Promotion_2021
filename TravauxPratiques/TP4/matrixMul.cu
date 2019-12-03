/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 */

// System includes
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <assert.h>
# ifdef WIN32
# include <time.h>
# else
# include <sys/time.h>
# endif

// CUDA runtime
#include <cuda_runtime.h>

/* Small function returning time in seconds. The
   precision is in micro-second order.
*/
double topChrono()
{
# ifdef WIN32
  clock_t chrono;
  chrono = clock();
  return ((double)chrono)/CLOCKS_PER_SEC;
# else
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return tv.tv_sec+1.E-6*tv.tv_usec;
# endif
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int nb_cols_A, int nb_cols_B)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = nb_cols_A * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + nb_cols_A - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * nb_cols_B;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + nb_cols_A * ty + tx];
        Bs[ty][tx] = B[b + nb_cols_B * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = nb_cols_B * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + nb_cols_B * ty + tx] = Csub;
}

void compute_tensor_vectors(int nu, float *u, int nv, float *v, float phase )
{
  const float pi = 3.141592653589793;

  for (int i = 0; i < nu; ++i )
    u[i] = std::cos((2*pi*i+phase)/nu);
  for ( int j = 0; j < nv; ++j )
    v[j] = std::sin((2*pi*j-phase)/nv);
}

void init_matrix(int nu, const float *u, int nv, const float *v, float *matrix)
{

  for (int i = 0; i < nu; ++i )
    for ( int j = 0; j < nv; ++j )
        matrix[i*nv + j] = u[i]*v[j];
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int block_size, dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // Initialize host memory
    const float phase_A = 0.3f;
    const float phase_B = 0.01f;
    float *h_uA = (float*)malloc(dimsA.y*sizeof(float));
    float *h_vA = (float*)malloc(dimsA.x*sizeof(float));
    compute_tensor_vectors(dimsA.y, h_uA, dimsA.x, h_vA, phase_A);
    init_matrix(dimsA.y, h_uA, dimsA.x, h_vA, h_A);
    float *h_uB = (float*)malloc(dimsB.y*sizeof(float));
    float *h_vB = (float*)malloc(dimsB.x*sizeof(float));
    compute_tensor_vectors(dimsB.y, h_uB, dimsB.x, h_vB, phase_B);
    init_matrix(dimsB.y, h_uB, dimsB.x, h_vB, h_B);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    cudaError_t error;
    double t1 = topChrono();
    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    double t2 = topChrono();
    printf("Time to allocate matrices : %g s\n",t2-t1);
    // copy host memory to device
    t1 = topChrono();
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    t2 = topChrono();
    printf("Time to copy A and B in GPU global memory : %g\n",t2-t1);
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    t1 = topChrono();
    if (block_size == 16)
    {
        matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else
    {
        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    cudaDeviceSynchronize();
    t2 = topChrono();

    printf("Time to compute C = A*B : %g\n",t2-t1);

    // Compute and print the performance
    double secPerMatrixMul = t2-t1;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / secPerMatrixMul;
    printf(
        "Performance= %.2g GFlop/s, Time= %.3g sec, Nb Flops= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops, secPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("Checking computed result for correctness using extern product: ");
    bool correct = true;
    float* h_c2 = (float*)calloc(dimsC.x*dimsC.y,sizeof(float));
    t1 = topChrono();
    // Calcul du produit scalaire (h_vA|h_uB)
    float scal = 0.f;
    for ( int i = 0; i < dimsA.x; ++i)
        scal += h_vA[i]*h_uB[i];
    for ( int i = 0; i < dimsC.y; ++i )
        for ( int j = 0; j < dimsC.x; ++j )
        {
            h_c2[i*dimsC.x+j] = scal*h_uA[i]*h_vB[j];
        }
    t2 = topChrono();
    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
      if (fabs(h_C[i] - h_c2[i]) > 1e-3/std::max(fabs(h_c2[i]), 1.E-6f))
        {
	  printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-5\n", i, h_C[i], h_c2[i]);
	  correct = false;
        }
    }    
    printf("%s\n", correct ? "OK" : "FAIL");
    secPerMatrixMul = t2-t1;
    flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / secPerMatrixMul;
    printf("Blas Performance= %.2g GFlop/s, Time= %.3g sec, Nb Flops= %.0f Ops\n",
        gigaFlops, secPerMatrixMul, flopsPerMatrixMul);
    free(h_c2);
    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNote: For peak performance, please refer to the matrixMulCUBLAS example.\n");
    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    // Use a larger block size for Fermi and above
    int block_size = 32; /* Try 32 for more recent GPU cards */

    dim3 dimsA(100*block_size, 100*block_size, 1);
    dim3 dimsB(200*block_size, 100*block_size, 1);


    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(block_size, dimsA, dimsB);

    exit(matrix_result);
}
