#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

typedef unsigned int hist_t;

__device__ uchar4 rgbToGray(uchar4 pixel)
{
    char gray = static_cast<char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    return make_uchar4(gray, gray, gray, pixel.w);
}

__global__ void rgb2grayKernel(const uchar4 *input, uchar4 *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = y * width + x;
        output[index] = rgbToGray(input[index]);
    }
}

__global__ void calcHistogramKernel(const uchar4 *image, hist_t *histogram, int width, int height)
{
    // Calculate thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate histogram for each pixel handled by one thread
    while (idx < width * height)
    {
        // Get the intensity value of the current pixel (assuming it's already grayscale)
        unsigned char intensity = image[idx].x;

        // Increment the corresponding value in the histogram vector
        atomicAdd(&histogram[intensity], 1);

        // Move to the next pixel
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void plotHistogramKernel(uchar4 *image, hist_t *histogram, int width, int height, int max_freq)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    uchar4 white_pixel = make_uchar4(255, 255, 255, 255);
    uchar4 black_pixel = make_uchar4(0, 0, 0, 255);
    if (index < 256)
    {
        int freq = histogram[index] * 256 / max_freq;
        for (int i = 0; i < 256; i++)
        {
            int row = height - i - 1;
            if (i <= freq)
            {
                image[row * width + 2 * index] = white_pixel;
                image[row * width + 2 * index + 1] = white_pixel;
            }
            else
            {
                image[row * width + 2 * index].x >>= 2;
                image[row * width + 2 * index].y >>= 2;
                image[row * width + 2 * index].z >>= 2;
                image[row * width + 2 * index + 1].x >>= 2;
                image[row * width + 2 * index + 1].y >>= 2;
                image[row * width + 2 * index + 1].z >>= 2;
            }
        }
    }
}




__global__ void calcHistogramSharedKernel(const uchar4 *image, hist_t *histogram, int width, int height)
{
    // Define the local histogram vector in shared memory
    __shared__ hist_t histo_local[256];

    // Initialize local histogram
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
    {
        histo_local[i] = 0;
    }

    // Synchronize threads to make sure the initialization is complete
    __syncthreads();

    // Calculate local histogram
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < width * height)
    {
        unsigned char intensity = image[idx].x;
        atomicAdd(&histo_local[intensity], 1);
        idx += blockDim.x * gridDim.x;
    }

    // Synchronize threads to make sure all threads have finished updating the local histogram
    __syncthreads();

    // Update the global histogram using atomic operations
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
    {
        atomicAdd(&histogram[i], histo_local[i]);
    }
}





int main(int argc, char **argv)
{
    // create input/output streams
    videoSource *input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput *outputOriginal = videoOutput::Create(argc, argv, ARG_POSITION(1));
    videoOutput *outputGray = videoOutput::Create(argc, argv, ARG_POSITION(2));

    if (!input || !outputOriginal || !outputGray)
        return 0;

    int width = input->GetWidth();
    int height = input->GetHeight();

    // Allocate memory for input and output buffers
    uchar4 *d_output;
    cudaMalloc(&d_output, width * height * sizeof(uchar4));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    hist_t *d_histogram;
    cudaMalloc(&d_histogram, 256 * sizeof(hist_t));
    int h_histogram[256];

    while (true)
    {
        // Capture frame to host buffer
        uchar4 *image = NULL;
        int status = 0;
        if (!input->Capture(&image, 1000, &status))
        {
            if (status == videoSource::TIMEOUT)
                continue;
            break; // EOS
        }

        // Launch the kernel
        rgb2grayKernel<<<gridSize, blockSize>>>(image, d_output, width, height);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        cudaMemset(d_histogram, 0, 256 * sizeof(hist_t)); // Zero the histogram vector
        //calcHistogramKernel<<<4, 256>>>(d_output, d_histogram, width, height);
       
        calcHistogramSharedKernel<<<4, 256>>>(d_output, d_histogram, width, height);
        cudaDeviceSynchronize();

        cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(hist_t), cudaMemcpyDeviceToHost);

        int idx = 0;
        int sum = 0;
        while (idx < 256)
        {
            sum += h_histogram[idx];
            idx++;
        }

        printf("%d\n", sum);

        // Display the original image
        // outputOriginal->Render(image, width, height);

        dim3 block(256, 1, 1);
        dim3 grid(1, 1, 1);
        plotHistogramKernel<<<1, 256>>>(d_output, d_histogram, width, height, 20000);

        // Display the converted grayscale image
        outputGray->Render(d_output, width, height);

        // Update status bar
        char str[256];
        sprintf(str, "Camera Viewer (%ux%u) | %0.1f FPS", width, height, outputGray->GetFrameRate());
        outputGray->SetStatus(str);

        if (!outputGray->IsStreaming()) // check if the user quit
            break;
    }

    // Free allocated memory
    cudaFree(d_output);
    cudaFree(d_histogram);

    return 0;
}
