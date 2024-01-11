// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  
  unsigned int tx = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  int stride = 1;
  
  if((start + 2*tx) < len)
  	T[2*tx] = input[start + 2*tx];
  else
  	T[2*tx] = 0;

  if((start + 2 * tx + 1) < len)
  	T[2 * tx + 1] = input[start + 2 * tx + 1];
  else
  	T[2 * tx + 1] = 0;
  
  
  // scan step
  while(stride < (BLOCK_SIZE*2)){
  	__syncthreads();
  	int idx = (tx + 1) * stride * 2 - 1;
  	if(idx < (BLOCK_SIZE*2) && (idx >= stride))
  		T[idx] += T[idx - stride];
  	stride *= 2;
  }
  
  // post scan step
  stride = BLOCK_SIZE/2;
  while(stride > 0){
  	__syncthreads();
  	int idx = (tx + 1) * stride * 2 - 1;
  	if((idx + stride) < (BLOCK_SIZE*2))
  		T[idx + stride] += T[idx];
  	stride /= 2;
  }
  
  
  __syncthreads();
  // copy back to global memory
  if((start + 2*tx) < len)
  	input[start + 2*tx] = T[2*tx];
  if((start + 2*tx + 1) < len)
  	input[start + 2*tx + 1] = T[2 * tx + 1];
  	
  if(len > (BLOCK_SIZE * 2) && (threadIdx.x == 0))
  	output[blockIdx.x] = T[2*BLOCK_SIZE - 1];

  
}

__global__ void add(float *input, float *output, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if((blockIdx.x > 0) && (len > (BLOCK_SIZE*2)) && (idx < len))
    input[idx] += output[blockIdx.x - 1];
}
  

  	
  
  

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  int numBlocks;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
        
  numBlocks = numElements / (BLOCK_SIZE << 1);
  if(numBlocks == 0 || numBlocks % ( BLOCK_SIZE << 1)) numBlocks++;
  

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numBlocks, 1, 1);
  dim3 SingleDimGrid(1, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 AddDimBlock(BLOCK_SIZE << 1, 1, 1);
  
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  scan<<<SingleDimGrid, DimBlock>>>(deviceOutput , NULL, numBlocks);
  add<<<DimGrid, AddDimBlock>>>(deviceInput, deviceOutput, numElements);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
