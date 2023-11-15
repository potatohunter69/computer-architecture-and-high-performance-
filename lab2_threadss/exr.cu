#include <iostream>
#include <math.h>

__global__ void multKernel(int n, float* a, float* b, float* c)
{
    int index = threadIdx.x;  // Get the thread's unique index within the block.
    int stride = blockDim.x;  // Get the total number of threads in the block.

    // Each thread processes a different set of elements in the array.
    for (int i = index; i < n; i += stride) {
        c[i] = a[i] * b[i];
    }

    // we shift the reads in each operation untill we reach the end 
}

int main()
{
    int N = 1 << 24; // 1 is 0000001 and <<24 shifts the 1 with 24 positons, which is 1000000...(24 0s) 
    float* h_a, *h_b, *h_c;
    float* d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];

    // Allocate device memory
    cudaMallocManaged(&d_a, N * sizeof(float));
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_a[i] = 2.0f;
        h_b[i] = 3.0f;
    }

    // Copy data from host to device
  //  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
   int blockSize = 256; // Choose an appropriate block size
   int numBlocks = (N + blockSize - 1) / blockSize; // Calculate the number of blocks needed
   multKernel<<<1, blockSize>>>(N, d_a, d_b, d_c);

   /*
   blockSize and numBlocks: these parameters determine the total number of threads running in prallel and how they are grouped into thread blocks




   */

   // Copy result back to host
   cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

   // Check result for errors (all values should be 6.0f)
   float maxError = 0.0f;
   for (int i = 0; i < N; i++)
       maxError = fmax(maxError, fabs(h_c[i] - 6.0f));

   std::cout << "Max error: " << maxError << std::endl;

   // Clean up
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);
   delete[] h_a;
   delete[] h_b;
   delete[] h_c;

   return 0;
}
