#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

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

    while (true) {
        // Capture frame to host buffer
        uchar4 *image = NULL;
        int status = 0;
        if (!input->Capture(&image, 1000, &status)) {
            if (status == videoSource::TIMEOUT)
                continue;
            break; // EOS
        }

        // Launch the kernel
        rgb2grayKernel<<<gridSize, blockSize>>>(image, d_output, width, height);
        cudaDeviceSynchronize();  // Wait for the kernel to finish

        // Display the original image
        outputOriginal->Render(image, width, height);

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
