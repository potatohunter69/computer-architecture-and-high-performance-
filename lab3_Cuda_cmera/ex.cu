#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>


/*

// blockDim = number of threads in a block
// blockId = the id of the block 
// int index = blockId.x * blockDIm.x + threadId.x  // with this we find a specefik thread 
// int stride = gridDim.x * blockDim.x  ; with this we switch to the next wrap 
// int blockNum =  (N + blockSize -1)/blockSize; // this used to calcualte the number of blocks we need, we could say N/blockSize , but 
// we use +256-1 to make sure we have enought blocks 

__device__ char4 Gray(char4 pixel){
    char gray = static_cast<char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z); 
    return make_char4(gray, gray, gray, )
}


__global__ void rgbToGray(uchar4 imgae, int width, int height){
    int index = blockId.x * blockDIm.x + threadId.x  
    int stride = gridDim.x * blockDim.x 

    for(int i = index; i< width*height; i += stride){
        image[i] = Gray
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
        
        int width = input->Getwidth();
        int height = input->GetHeight();
        int N = width * height;  
        int blockSize = 256; 
        int blockNum =  (N + blockSize -1)/blockSize; 
        rgbToGrayKernal<<<blockNum, blockSize>>>(image, width, height)

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


************ having diffrent stream and buffer for original picutre and modified 
// instead of rewriting the pciture we can allcate a new buffer and save the modification in that buffer, and have two diffrent 
streams for showing the orginal and modyfied one, om man använder, 

__device__ char4 Gray(char4 pixel){
    char gray = static_cast<char>(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z); 
    return make_char4(gray, gray, gray, )
}


__global__ void rgbToGray(uchar4 input, uchar4 output, int width, int height){
    int index = blockId.x * blockDIm.x + threadId.x  
    int stride = gridDim.x * blockDim.x 

    for(int i = index; i< width*height; i += stride){
        output[i] = Gray(input[i])
    }
}


int main(int argc, char **argv)
{
    // create input/output streams
    videoSource *input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput *output = videoOutput::Create(argc, argv, ARG_POSITION(1));
    videoOutput *outputGray = videoOutput::Create(argc, argv, ARG_POSITION(2));

    
    uchar4 d_output; 
    cudaMalloc(&d_ouput, width*height*sizeof(uchar4)); 

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
        
        int width = input->Getwidth();
        int height = input->GetHeight();
        int N = width * height;  
        int blockSize = 256; 
        int blockNum =  (N + blockSize -1)/blockSize; 
        rgbToGrayKernal<<<blockNum, blockSize>>>(image, d_output,width, height)

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


*/