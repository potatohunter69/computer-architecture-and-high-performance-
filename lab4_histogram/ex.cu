#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

typedef unsigned int hist_t;

__device__ uchar4 rgbToGray(uchar4 pixel) {
    char gray = static_cast<char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    return make_uchar4(gray, gray, gray, pixel.w);
}

__global__ void rgb2grayKernel(const uchar4 *input, uchar4 *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        output[index] = rgbToGray(input[index]);
    }
}


__global__ void callHistogram(uchar4* image, hist_t *histogram, int width, int height){
    int index = blockDim.x * blockIdx.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 

 
    for(int i = index; i<width*height; i += stride){
        char intesity = image[i].x;  
        atomicAdd(&histogram[intesity], 1); 
    }
}

__global__ void calHistogramShared(uchar4* image, hist_t *histogram, int width, int height){
    __shared__ hist_t local_histogram[256];

    if(threadIdx.x < 256){
        local_histogram[threadIdx.x] = 0; 
    }
    
    __syncthreads(); 

    int index = blockDim.x * blockIdx.x + threadIdx.x; 
    while(index < width* height){
        char intensity = image[index].x; 
        atomicAdd(&local_histogram[intensity], 1); 
        index += blockDim.x * gridDim.x; 

    }
    
    __syncthreads(); 


    if(threadIdx.x < 256){
        atomicAdd(&histogram[threadIdx.x], local_histogram[threadIdx.x]);  
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


int main(int argc, char **argv) {
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

    hist_t* d_histogram; 
    cudaMalloc(&d_histogram, 256*sizeof(hist_t));
    int h_histogram[256]; 

    while (true) {
        // Capture frame to host buffer
        uchar4 *image = NULL;
        int status = 0;
        if (!input->Capture(&image, 1000, &status)) {
            if (status == videoSource::TIMEOUT)
                continue;
            break; // EOSkDim.y + threadIdx.y;

        }

        // Launch the kernel
        rgb2grayKernel<<<gridSize, blockSize>>>(image, d_output, width, height);
        cudaDeviceSynchronize();  // Wait for the kernel to finish

        cudaMemset(d_histogram, 0, 256*sizeof(hist_t)); 
       //callHistogram<<<4, 256>>>(d_output, d_histogram, width, height); 
        calHistogramShared<<<4,256>>>(d_output, d_histogram, width, height); 

        cudaMemcpy(h_histogram, d_histogram, 256*sizeof(hist_t), cudaMemcpyDeviceToHost); 

        int sum = 0; 
        int idx = 0; 
        while(idx < 256){
            sum += h_histogram[idx]; 
            idx++; 
        }
        
        printf("%d \n", sum); 

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

    return 0;
}
