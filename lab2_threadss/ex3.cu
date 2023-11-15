#include <iostream>
#include <math.h>

__global__ void multKernel(int n, float* a, float* b, float* c, int* perm)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        int idx = perm[i];
        c[idx] = a[idx] * b[idx];
    }
}


int main() {
    int N = 1 << 24; // 1 is 0000001 and <<24 shifts the 1 with 24 positons, which is 1000000...(24 0s) 
    float* h_a, *h_b, *h_c;
    float* d_a, *d_b, *d_c;
    int *h_perm, *d_perm; // Add h_perm and d_perm

    // Allocate host memory
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    h_perm = new int[N]; 

    // Allocate device memory
    cudaMalloc(&d_perm, N * sizeof(int));

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));


    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_a[i] = 2.0f;
        h_b[i] = 3.0f;
        h_perm[i] = i;
    }

   for (int i = N-1; i > 0; i--)
   {
      int j = rand() % (i + 1);
      std::swap(h_perm[i], h_perm[j]);
   }

    cudaMemcpy(d_perm, h_perm, N * sizeof(int), cudaMemcpyHostToDevice); // Copy h_perm to d_perm

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
   int blockSize = 256; // Choose an appropriate block size
   int numBlocks = (N + blockSize - 1) / blockSize; // Calculate the number of blocks needed
   multKernel<<<numBlocks, blockSize>>>(N, d_a, d_b, d_c, d_perm);

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
    cudaFree(d_perm);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_perm;

    return 0;
}



