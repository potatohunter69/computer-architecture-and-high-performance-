__global__ void calHistogram(uchar4* image, hist_t* histogram, int width, int height){
    int index = blockDim.x * blockID.x + threadId.x;
    int stride = blockDim.x * gridDim.x 

    while(index < widht * height){
        uchar1 intensity = histogram[index].x
        histogram[intesity]++; 
        index += stride 
    }

}


int main(void){

    cudamemset(histogram, 0, fsdj)

    calHistogram<<<4,256>>>(image, histogram, width, height)

    cudamemcpy(h_histogram, d_istogram)
     
}