#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

__device__ uchar4 rgbToGray(uchar4 pixel) {
    char gray = static_cast<char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    return make_uchar4(gray, gray, gray, pixel.w);
}

__global__ void rgb2grayKernel(uchar4 *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        image[index] = rgbToGray(image[index]);
    }
}


int main(int argc, char **argv)
{
    // create input/output streams
    videoSource *input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput *output = videoOutput::Create(argc, argv, ARG_POSITION(1));
    if (!input)
        return 0;
    // capture/display loop
    while (true)
    {
        uchar4 *image = NULL;                       // can be uchar3, uchar4, float3, float4
        int status = 0;                             // see videoSource::Status (OK, TIMEOUT, EOS,ERROR)
        if (!input->Capture(&image, 1000, &status)) // 1000ms timeout (default)
        {
            if (status == videoSource::TIMEOUT)
                continue;
            break; // EOS
        }

        int width = input->GetWidth();
        int height = input->GetHeight();
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        rgb2grayKernel<<<gridSize, blockSize>>>(image, width, height);
        cudaDeviceSynchronize();  // Wait for the kernel to finish

        if (output != NULL)
        {
            output->Render(image, input->GetWidth(), input->GetHeight());
            // Update status bar
            char str[256];
            sprintf(str, "Camera Viewer (%ux%u) | %0.1f FPS", input->GetWidth(),
                    input->GetHeight(), output->GetFrameRate());
            output->SetStatus(str);
            if (!output->IsStreaming()) // check if the user quit
                break;
        }
    }
}



