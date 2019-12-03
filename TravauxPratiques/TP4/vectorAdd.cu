/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. 
 */

#include <stdio.h>
# ifdef WIN32
# include <time.h>
# else
# include <sys/time.h>
# endif

// For the CUDA runtime routines (prefixed with "cuda_")
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
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *u, const float *v, float *w, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        w[i] = u[i] + v[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector u
    float *h_u = (float *)malloc(size);

    // Allocate the host input vector v
    float *h_v = (float *)malloc(size);

    // Allocate the host output vector w
    float *h_w = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_u == NULL || h_v == NULL || h_w == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_u[i] = rand()/(float)RAND_MAX;
        h_v[i] = rand()/(float)RAND_MAX;
    }

    double t1 = topChrono();
    // Allocate the device input vector u
    float *d_u = NULL;
    err = cudaMalloc((void **)&d_u, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector u (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector v
    float *d_v = NULL;
    err = cudaMalloc((void **)&d_v, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector v (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_w = NULL;
    err = cudaMalloc((void **)&d_w, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    double t2 = topChrono();
    printf("Time to allocate gpu arrays : %g s\n",t2-t1);
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    t1 = topChrono();
    err = cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector u from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    t2 = topChrono();
    printf("Time to copy input data from host to GPU device : %g s\n", t2-t1);
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    t1 = topChrono();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_u, d_v, d_w, numElements);
    t2 = topChrono();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Time to execute CUDA kernel : %g s\n", t2-t1);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    t1 = topChrono();
    err = cudaMemcpy(h_w, d_w, size, cudaMemcpyDeviceToHost);
    t2 = topChrono();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector w from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Time to copy from GPU device to host : %g s\n",t2-t1);
    // Verify that the result vector is correct
    float* h_w2 = (float*)malloc(numElements*sizeof(float));

    t1 = topChrono();
    for (int i = 0; i < numElements; ++i)
    {
      h_w2[i] = h_u[i] + h_v[i];
    }
    t2 = topChrono();
    printf("Time to add vectors on CPU : %g\n", t2-t1);

    for (int i = 0; i < numElements; ++i)
      if (fabs(h_w[i]-h_w2[i]) > 1e-5) {
	fprintf(stderr, "Result verification failed at element %d!\n", i);
	exit(EXIT_FAILURE);
      }

    // Free device global memory
    err = cudaFree(d_u);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector u (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_v);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector v (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_w);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_u);
    free(h_v);
    free(h_w);

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

