
#include <wb.h>
#include <cuda_runtime_api.h>


//Define TileWith
#define TILEWIDTH 1
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
                               
  
  __shared__ float subTileA[TILEWIDTH][TILEWIDTH];
  __shared__ float subTileB[TILEWIDTH][TILEWIDTH];
  
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  //@@ Insert code to implement matrix multiplication here
  int Row = by * TILEWIDTH + ty;
  int Col = bx * TILEWIDTH + tx;  
  float pval = 0;
  for(int q = 0 ; q < numAColumns/TILEWIDTH ; ++q){
  	subTileA[ty][tx] = A[Row * numAColumns + (q * TILEWIDTH + tx)];
  	subTileB[ty][tx] = B[(q * TILEWIDTH + ty) * numBColumns + Col];
	__syncthreads();
	for(int k = 0; k < TILEWIDTH ; ++k){
		pval += subTileA[ty][k] * subTileB[k][tx];
	}
	__syncthreads();
	C[Row * numCColumns + Col] = pval;
  }
  
  /*
  int Row = by * TILEWIDTH + ty;
  int Col = bx * TILEWIDTH + tx;  
  int RowIn = Row - TILEWIDTH/2;
  int ColIn = Row - TILEWIDTH/2;
  float pval = 0;
  for(int q = 0 ; q < numAColumns/TILEWIDTH ; ++q){
  	if((RowIn >= 0) && (RowIn < numAColumns) && 
  	    ColIn >= 0) && (ColIn < numBColumns))
  	subTileA[ty][tx] = A[Row * numAColumns + (q * TILEWIDTH + tx)];
  	subTileB[ty][tx] = B[(q * TILEWIDTH + ty) * numBColumns + Col];
	__syncthreads();
	for(int k = 0; k < TILEWIDTH ; ++k){
		pval += subTileA[ty][k] * subTileB[k][tx];
	}
	__syncthreads();
	C[Row * numCColumns + Col] = pval;
  }
  */
  
  /*
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if((Row < numCRows) && (Col < numCColumns)){
  	float p = 0;
  	for (int k = 0 ; k < numAColumns ; ++k){
  		p += A[Row*numAColumns + k] * B[k*numBRows + Col];
  	}
  	C[Row*numCColumns + Col] = p;
  } 
  */
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns ...Ok
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix ...OK
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
                            
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here ...OK
  cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float)); 
  cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here ...OK
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here ...OK
  dim3 dimGrid(ceil((1.0*numCColumns)/TILEWIDTH), ceil((1.0*numCRows)/TILEWIDTH), 1);
  dim3 dimBlock(TILEWIDTH, TILEWIDTH , 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here ...OK
  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows , numAColumns, numBRows , numBColumns, numCRows , numCColumns);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here ...OK
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here ...OK
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
