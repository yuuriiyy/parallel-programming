// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void float2uchar(float *input, unsigned char *output, int width, int height){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;
  int idx = (width * height) * bz + width * y + x;
  if(y < height && x < width) output[idx] = (unsigned char) 255*input[idx];
}

__global__ void rgb2gray(unsigned char *input, unsigned char *output, int width, int height){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;
  int idx = y * width + x;
  int r, g, b;
  if(x < width && y < height){
    r = input[3*idx];
    g = input[3*idx + 1];
    b = input[3*idx + 2];
    output[idx] = (unsigned char) (0.299*r + 0.587*g + 0.114*b);
  }
}

__global__ void gray2hist(unsigned char *input, unsigned int *output, int width, int height){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;  
  int ty = threadIdx.y;
  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;
  __shared__ unsigned int histogram[HISTOGRAM_LENGTH];
  int tIdx = blockDim.x * ty + tx;
  if (tIdx < HISTOGRAM_LENGTH)
    histogram[tIdx] = 0;
  
  __syncthreads();
  
  if (x < width && y < height) {
    int idx = y * width + x;
    unsigned char val = input[idx];
    atomicAdd(&(histogram[val]), 1);
    // if (val < HISTOGRAM_LENGTH) 
    //   atomicAdd(&(output[val]), 1);
  }

  __syncthreads();
  if (tIdx < HISTOGRAM_LENGTH) 
    atomicAdd(&(output[tIdx]), histogram[tIdx]);
}

__global__ void hist2CDF(unsigned int *input, float *output, int width, int height){
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  
  if(tx < HISTOGRAM_LENGTH)
    cdf[tx] = input[tx];
  __syncthreads();
  
  //first scan
  int stride = 1;
  while(stride < HISTOGRAM_LENGTH){
        __syncthreads();
        int idx = (tx + 1)*stride*2 - 1;
        if(idx < HISTOGRAM_LENGTH && (idx-stride) >= 0)
            cdf[idx] += cdf[idx-stride];
        stride = stride * 2;
  }
  //post scan
  stride = HISTOGRAM_LENGTH/4;
  while(stride > 0){
        __syncthreads();
        int idx = (tx + 1)*stride*2 - 1;
        if((idx + stride) < HISTOGRAM_LENGTH)
	        cdf[idx + stride] += cdf[idx];				
        stride = stride / 2;
  }
  //calculate probability P(x)
  __syncthreads();
  output[tx] = cdf[tx] / ((float)(width * height));
}

__global__ void equal(unsigned char *img, float *cdf, int width, int height){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z; 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;
  
  if(x < width && y < height){
    int idx = (width * height) * bz + (width) * y + x;
    float temp = 255*(cdf[img[idx]] - cdf[0])/(1.0 - cdf[0]);
    img[idx] = (unsigned char) min(max(temp, 0.0), 255.0);
  }
}

__global__ void uchar2float(unsigned char *input, float *output, int width, int height){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z; 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;
  if(x < width && y < height){
    int idx = (width * height)*bz + (width)*y + x;
    output[idx] = (input[idx] / 255.0);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *imageFloat;
  unsigned char *imageUchar;
  unsigned char *imageGray;
  unsigned int *imageHist;
  float *imageCDf; 
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  
  int imageSize = imageWidth * imageHeight;
  
  //@@ insert code here
  cudaMalloc((void**) &imageFloat, imageSize * imageChannels * sizeof(float)); // need to define imageFloat
  cudaMalloc((void**) &imageUchar, imageSize * imageChannels * sizeof(float));  // need to define imageUchar
  cudaMalloc((void**) &imageGray, imageSize * 1 * sizeof(float));              // number of channel = 1
  cudaMalloc((void**) &imageHist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void**) imageHist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));  // cudaError_t cudaMemset(void* devPtr, int value, size_t count)
                                                                                // initializes or sets device memory to a value
  cudaMalloc((void**) &imageCDf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(imageFloat, hostInputImageData, imageSize*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 dimGrid(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dim3 dimBlock(32, 32, 1);
  float2uchar<<<dimGrid, dimBlock>>>(imageFloat, imageUchar, imageWidth, imageHeight);
  cudaDeviceSynchronize();
 
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  rgb2gray<<<dimGrid, dimBlock>>>(imageUchar, imageGray, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  gray2hist<<<dimGrid, dimBlock>>>(imageGray, imageHist, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  hist2CDF<<<dimGrid, dimBlock>>>(imageHist, imageCDf, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  equal<<<dimGrid, dimBlock>>>(imageUchar, imageCDf, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  uchar2float<<<dimGrid, dimBlock>>>(imageUchar, imageFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();                                                                              
                                                                                
                                                                                
  cudaMemcpy(hostOutputImageData, imageFloat, imageSize*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(imageFloat);
  cudaFree(imageUchar);
  cudaFree(imageGray);
  cudaFree(imageHist);
  cudaFree(imageCDf);
  
  return 0;
}
