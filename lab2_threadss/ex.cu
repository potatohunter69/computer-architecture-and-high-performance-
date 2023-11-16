#include <iostream>
#include <math.h>

/*

******************* with only one thread



__global__ void calculateKernal(float *d_a, float *d_b, float *d_c, int N)
{

    for (int i = 0; i < N; i++)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void)
{
    int N = 1 << 24;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // allocate for host
    h_a  = new float[N];
    h_b  = new float[N];
    h_c  = new float[N];

    // allocate for device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // initial values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 0.1;
        h_b[i] = 0.1;
    }

    // copy host data to device data
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // start the keranl
    int blocksize = 1;
    int numBlock = 1;
    calculateKernal<<<numBlock, blocksize>>>(d_a, d_b, d_c, N);

    // copy back the  result from device back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_c[i] - 6.0f));

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}


************************** with more threads and blocks

// more blocks doesent always mean better perfomance and it dpencs on the data and size of data

__global__ void calculateKernal(float *d_a, float *d_b, float *d_c, int N)
{
    int index = threadIdx.x * blockDim.x + threadIdx.x;
    int blockNum = blockDim.x * gridDim.x;

    for (int i = index ; i < N; i += blockNum)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void)
{
    int N = 1 << 24;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // allocate for host
    h_a  = new float[N];
    h_b  = new float[N];
    h_c  = new float[N];

    // allocate for device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // initial values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 0.1;
        h_b[i] = 0.1;
    }

    // copy host data to device data
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // start the keranl
    int blocksize = 256;
    int numBlock = (N + blocksize -1)/ blocksize;
    calculateKernal<<<numBlock, blocksize>>>(d_a, d_b, d_c, N);

    // copy back the  result from device back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_c[i] - 0.2f));

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}



************************** with more threads

// although more threads means better perfomance, but more more therads doesn't alway mean better, since gpu resouces is limited and if we exceed this limit
this can couse worse perfomce and other problems

__global__ void calculateKernal(float *d_a, float *d_b, float *d_c, int N)
{
    int index = threadIdx.x;   /// * blockDim.x + threadIdx.x;
    int blockNum = blockDim.x;  // * gridDim.x;

    for (int i = index ; i < N; i += blockNum)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void)
{
    int N = 1 << 24;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // allocate for host
    h_a  = new float[N];
    h_b  = new float[N];
    h_c  = new float[N];

    // allocate for device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // initial values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 0.1;
        h_b[i] = 0.1;
    }

    // copy host data to device data
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // start the keranl
    int blocksize = 1024;
    int numBlock = (N + blocksize -1)/ blocksize;
    calculateKernal<<<1, blocksize>>>(d_a, d_b, d_c, N);

    // copy back the  result from device back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_c[i] - 0.2f));

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}



#include <iostream>
#include <math.h>

/*

******************* with only one thread



__global__ void calculateKernal(float *d_a, float *d_b, float *d_c, int N)
{

    for (int i = 0; i < N; i++)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void)
{
    int N = 1 << 24;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // allocate for host
    h_a  = new float[N];
    h_b  = new float[N];
    h_c  = new float[N];

    // allocate for device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // initial values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 0.1;
        h_b[i] = 0.1;
    }

    // copy host data to device data
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // start the keranl
    int blocksize = 1;
    int numBlock = 1;
    calculateKernal<<<numBlock, blocksize>>>(d_a, d_b, d_c, N);

    // copy back the  result from device back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_c[i] - 6.0f));

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}


************************** with more threads and blocks

// more blocks doesent always mean better perfomance and it dpencs on the data and size of data

__global__ void calculateKernal(float *d_a, float *d_b, float *d_c, int N)
{
    int index = threadIdx.x * blockDim.x + threadIdx.x;
    int blockNum = blockDim.x * gridDim.x;

    for (int i = index ; i < N; i += blockNum)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void)
{
    int N = 1 << 24;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // allocate for host
    h_a  = new float[N];
    h_b  = new float[N];
    h_c  = new float[N];

    // allocate for device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // initial values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 0.1;
        h_b[i] = 0.1;
    }

    // copy host data to device data
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // start the keranl
    int blocksize = 256;
    int numBlock = (N + blocksize -1)/ blocksize;
    calculateKernal<<<numBlock, blocksize>>>(d_a, d_b, d_c, N);

    // copy back the  result from device back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_c[i] - 0.2f));

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}



************************** with more threads

// although more threads means better perfomance, but more more therads doesn't alway mean better, since gpu resouces is limited and if we exceed this limit
this can couse worse perfomce and other problems

__global__ void calculateKernal(float *d_a, float *d_b, float *d_c, int N)
{
    int index = threadIdx.x;   /// * blockDim.x + threadIdx.x;
    int blockNum = blockDim.x;  // * gridDim.x;

    for (int i = index ; i < N; i += blockNum)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void)
{
    int N = 1 << 24;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    // allocate for host
    h_a  = new float[N];
    h_b  = new float[N];
    h_c  = new float[N];

    // allocate for device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // initial values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 0.1;
        h_b[i] = 0.1;
    }

    // copy host data to device data
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // start the keranl
    int blocksize = 1024;
    int numBlock = (N + blocksize -1)/ blocksize;
    calculateKernal<<<1, blocksize>>>(d_a, d_b, d_c, N);

    // copy back the  result from device back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_c[i] - 0.2f));

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}


******** when we randomize the memor the performance is poor because of lack of coalesced memory access, gpus are desinced for coalesced momry access , 
and read data with consecutive threadsa nad consecutev momry, the memory is randomize the perfomace is bad 

__global__ void calculateKernal(float *d_a, float *d_b, float *d_c, int N)
{
    int index = threadIdx.x * blockDim.x + threadIdx.x;
    int blockNum = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += blockNum)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(void)
{
    int N = 1 << 24;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;

    int *h_perm, *d_perm;

    // allocate for host
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    h_perm = new int[N];

    // allocate for device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMalloc(&d_perm, N * sizeof(int));

    // initial values
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 0.1;
        h_b[i] = 0.1;
        h_perm[i] = i;
    }

    for (int i = 0; i < N; i++)
    {
        int j = rand() % (i + 1);
        std::swap(h_perm[i], h_perm[j]);
    }

    // copy host data to device data
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // start the keranl
    int blocksize = 1024;
    int numBlock = (N + blocksize - 1) / blocksize;
    calculateKernal<<<numBlock, blocksize>>>(d_a, d_b, d_c, N);

    // copy back the  result from device back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(h_c[i] - 0.2f));

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

*/
