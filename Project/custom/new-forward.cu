#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h> 

#define TILE_WIDTH 12
#define Mask_Width 7
#define Channel 4
#define map_size 16
#define SM_IN (TILE_WIDTH + Mask_Width - 1)

__constant__ float CM [map_size * Channel * Mask_Width * Mask_Width]; // Using Constant memory for optimization

// Original Code
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // using constant memory

    // Insert your GPU convolution kernel code here
    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int h = (blockIdx.z / W_tile) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_tile) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;

    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if (h * S + p < H && w * S + q < W) {
                    acc += in_4d(bx, c, h*S+p, w*S+q) * mask_4d(by, c, p, q);
                }
            }
        }
    }
    if (h < H_out && w < W_out) {
        out_4d(bx, by, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_constant_mem(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    //#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // using constant memory

    // Insert your GPU convolution kernel code here
    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int h = (blockIdx.z / W_tile) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_tile) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;

    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if (h * S + p < H && w * S + q < W) {
                    acc += in_4d(bx, c, h*S+p, w*S+q) * mask_4d(by, c, p, q);
                }
            }
        }
    }
    if (h < H_out && w < W_out) {
        out_4d(bx, by, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_unroll(float *output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_tile = ceil(1.0 * W_out / TILE_WIDTH); // W_tile is number of tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int b = bx;
    int m = by;
    int w = (bz % W_tile) * TILE_WIDTH + tx;
    int h = (bz / W_tile) * TILE_WIDTH + ty;


    float acc = 0.0f;

    for (int c = 0; c < C; ++c) {
        if(K == 3){
            acc += in_4d(b, c, h*S+0, w*S+0) * mask_4d(m, c, 0, 0);
            acc += in_4d(b, c, h*S+0, w*S+1) * mask_4d(m, c, 0, 1);
            acc += in_4d(b, c, h*S+0, w*S+2) * mask_4d(m, c, 0, 2);

            acc += in_4d(b, c, h*S+1, w*S+0) * mask_4d(m, c, 1, 0);
            acc += in_4d(b, c, h*S+1, w*S+1) * mask_4d(m, c, 1, 1);
            acc += in_4d(b, c, h*S+1, w*S+2) * mask_4d(m, c, 1, 2);

            acc += in_4d(b, c, h*S+2, w*S+0) * mask_4d(m, c, 2, 0);
            acc += in_4d(b, c, h*S+2, w*S+1) * mask_4d(m, c, 2, 1);
            acc += in_4d(b, c, h*S+2, w*S+2) * mask_4d(m, c, 2, 2);
        }
        if (K > 3) {
            acc += in_4d(b, c, h*S+0, w*S+0) * mask_4d(m, c, 0, 0);
            acc += in_4d(b, c, h*S+0, w*S+1) * mask_4d(m, c, 0, 1);
            acc += in_4d(b, c, h*S+0, w*S+2) * mask_4d(m, c, 0, 2);                
            acc += in_4d(b, c, h*S+0, w*S+3) * mask_4d(m, c, 0, 3);
            acc += in_4d(b, c, h*S+0, w*S+4) * mask_4d(m, c, 0, 4);
            acc += in_4d(b, c, h*S+0, w*S+5) * mask_4d(m, c, 0, 5);
            acc += in_4d(b, c, h*S+0, w*S+6) * mask_4d(m, c, 0, 6);
        
            acc += in_4d(b, c, h*S+1, w*S+0) * mask_4d(m, c, 1, 0);
            acc += in_4d(b, c, h*S+1, w*S+1) * mask_4d(m, c, 1, 1);
            acc += in_4d(b, c, h*S+1, w*S+2) * mask_4d(m, c, 1, 2);
            acc += in_4d(b, c, h*S+1, w*S+3) * mask_4d(m, c, 1, 3);
            acc += in_4d(b, c, h*S+1, w*S+4) * mask_4d(m, c, 1, 4);
            acc += in_4d(b, c, h*S+1, w*S+5) * mask_4d(m, c, 1, 5);
            acc += in_4d(b, c, h*S+1, w*S+6) * mask_4d(m, c, 1, 6);
        
            acc += in_4d(b, c, h*S+2, w*S+0) * mask_4d(m, c, 2, 0);
            acc += in_4d(b, c, h*S+2, w*S+1) * mask_4d(m, c, 2, 1);
            acc += in_4d(b, c, h*S+2, w*S+2) * mask_4d(m, c, 2, 2);
            acc += in_4d(b, c, h*S+2, w*S+3) * mask_4d(m, c, 2, 3);
            acc += in_4d(b, c, h*S+2, w*S+4) * mask_4d(m, c, 2, 4);
            acc += in_4d(b, c, h*S+2, w*S+5) * mask_4d(m, c, 2, 5);
            acc += in_4d(b, c, h*S+2, w*S+6) * mask_4d(m, c, 2, 6);
        
            acc += in_4d(b, c, h*S+3, w*S+0) * mask_4d(m, c, 3, 0);
            acc += in_4d(b, c, h*S+3, w*S+1) * mask_4d(m, c, 3, 1);
            acc += in_4d(b, c, h*S+3, w*S+2) * mask_4d(m, c, 3, 2);
            acc += in_4d(b, c, h*S+3, w*S+3) * mask_4d(m, c, 3, 3);
            acc += in_4d(b, c, h*S+3, w*S+4) * mask_4d(m, c, 3, 4);
            acc += in_4d(b, c, h*S+3, w*S+5) * mask_4d(m, c, 3, 5);
            acc += in_4d(b, c, h*S+3, w*S+6) * mask_4d(m, c, 3, 6);
        
            acc += in_4d(b, c, h*S+4, w*S+0) * mask_4d(m, c, 4, 0);
            acc += in_4d(b, c, h*S+4, w*S+1) * mask_4d(m, c, 4, 1);
            acc += in_4d(b, c, h*S+4, w*S+2) * mask_4d(m, c, 4, 2);
            acc += in_4d(b, c, h*S+4, w*S+3) * mask_4d(m, c, 4, 3);
            acc += in_4d(b, c, h*S+4, w*S+4) * mask_4d(m, c, 4, 4);
            acc += in_4d(b, c, h*S+4, w*S+5) * mask_4d(m, c, 4, 5);
            acc += in_4d(b, c, h*S+4, w*S+6) * mask_4d(m, c, 4, 6);
        
            acc += in_4d(b, c, h*S+5, w*S+0) * mask_4d(m, c, 5, 0);
            acc += in_4d(b, c, h*S+5, w*S+1) * mask_4d(m, c, 5, 1);
            acc += in_4d(b, c, h*S+5, w*S+2) * mask_4d(m, c, 5, 2);
            acc += in_4d(b, c, h*S+5, w*S+3) * mask_4d(m, c, 5, 3);
            acc += in_4d(b, c, h*S+5, w*S+4) * mask_4d(m, c, 5, 4);
            acc += in_4d(b, c, h*S+5, w*S+5) * mask_4d(m, c, 5, 5);
            acc += in_4d(b, c, h*S+5, w*S+6) * mask_4d(m, c, 5, 6);
        
            acc += in_4d(b, c, h*S+6, w*S+0) * mask_4d(m, c, 6, 0);
            acc += in_4d(b, c, h*S+6, w*S+1) * mask_4d(m, c, 6, 1);
            acc += in_4d(b, c, h*S+6, w*S+2) * mask_4d(m, c, 6, 2);
            acc += in_4d(b, c, h*S+6, w*S+3) * mask_4d(m, c, 6, 3);
            acc += in_4d(b, c, h*S+6, w*S+4) * mask_4d(m, c, 6, 4);
            acc += in_4d(b, c, h*S+6, w*S+5) * mask_4d(m, c, 6, 5);
            acc += in_4d(b, c, h*S+6, w*S+6) * mask_4d(m, c, 6, 6);
        } 
    }
    if(h < H_out && w < W_out){
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_unroll_fp16(float *output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_tile = ceil(1.0 * W_out / TILE_WIDTH); // W_tile is number of tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int b = bx;
    int m = by;
    int w = (bz % W_tile) * TILE_WIDTH + tx;
    int h = (bz / W_tile) * TILE_WIDTH + ty;


    // float acc = 0.0f;
    __half acc = __float2half(0.0f);
    for (int c = 0; c < C; ++c) {
        if(K == 3){
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+0)), __float2half(mask_4d(by, c, 0, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+1)), __float2half(mask_4d(by, c, 0, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+2)), __float2half(mask_4d(by, c, 0, 2))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+0)), __float2half(mask_4d(by, c, 1, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+1)), __float2half(mask_4d(by, c, 1, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+2)), __float2half(mask_4d(by, c, 1, 2))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+0)), __float2half(mask_4d(by, c, 2, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+1)), __float2half(mask_4d(by, c, 2, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+2)), __float2half(mask_4d(by, c, 2, 2))));    
        }
        if (K > 3) {
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+0)), __float2half(mask_4d(by, c, 0, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+1)), __float2half(mask_4d(by, c, 0, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+2)), __float2half(mask_4d(by, c, 0, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+3)), __float2half(mask_4d(by, c, 0, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+4)), __float2half(mask_4d(by, c, 0, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+5)), __float2half(mask_4d(by, c, 0, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+6)), __float2half(mask_4d(by, c, 0, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+0)), __float2half(mask_4d(by, c, 1, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+1)), __float2half(mask_4d(by, c, 1, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+2)), __float2half(mask_4d(by, c, 1, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+3)), __float2half(mask_4d(by, c, 1, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+4)), __float2half(mask_4d(by, c, 1, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+5)), __float2half(mask_4d(by, c, 1, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+6)), __float2half(mask_4d(by, c, 1, 6))));


            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+0)), __float2half(mask_4d(by, c, 2, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+1)), __float2half(mask_4d(by, c, 2, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+2)), __float2half(mask_4d(by, c, 2, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+3)), __float2half(mask_4d(by, c, 2, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+4)), __float2half(mask_4d(by, c, 2, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+5)), __float2half(mask_4d(by, c, 2, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+6)), __float2half(mask_4d(by, c, 2, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+0)), __float2half(mask_4d(by, c, 3, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+1)), __float2half(mask_4d(by, c, 3, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+2)), __float2half(mask_4d(by, c, 3, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+3)), __float2half(mask_4d(by, c, 3, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+4)), __float2half(mask_4d(by, c, 3, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+5)), __float2half(mask_4d(by, c, 3, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+6)), __float2half(mask_4d(by, c, 3, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+0)), __float2half(mask_4d(by, c, 4, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+1)), __float2half(mask_4d(by, c, 4, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+2)), __float2half(mask_4d(by, c, 4, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+3)), __float2half(mask_4d(by, c, 4, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+4)), __float2half(mask_4d(by, c, 4, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+5)), __float2half(mask_4d(by, c, 4, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+6)), __float2half(mask_4d(by, c, 4, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+0)), __float2half(mask_4d(by, c, 5, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+1)), __float2half(mask_4d(by, c, 5, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+2)), __float2half(mask_4d(by, c, 5, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+3)), __float2half(mask_4d(by, c, 5, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+4)), __float2half(mask_4d(by, c, 5, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+5)), __float2half(mask_4d(by, c, 5, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+6)), __float2half(mask_4d(by, c, 5, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+0)), __float2half(mask_4d(by, c, 6, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+1)), __float2half(mask_4d(by, c, 6, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+2)), __float2half(mask_4d(by, c, 6, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+3)), __float2half(mask_4d(by, c, 6, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+4)), __float2half(mask_4d(by, c, 6, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+5)), __float2half(mask_4d(by, c, 6, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+6)), __float2half(mask_4d(by, c, 6, 6))));
        } 
    }
    if(h < H_out && w < W_out){
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_fp16(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // using constant memory

    // Insert your GPU convolution kernel code here
    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int h = (blockIdx.z / W_tile) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_tile) * TILE_WIDTH + threadIdx.x;
    // float acc = 0.0f;
    __half acc = __float2half(0.0f);


    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if (h * S + p < H && w * S + q < W) {
                    // acc += in_4d(bx, c, h*S+p, w*S+q) * mask_4d(by, c, p, q);
                    acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+p, w*S+q)), __float2half(mask_4d(by, c, p, q))));
                }
            }
        }
    }
    if (h < H_out && w < W_out) {
        out_4d(bx, by, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_unroll_atomicadd(float *output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_tile = ceil(1.0 * W_out / TILE_WIDTH); // W_tile is number of tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int w = (bz % W_tile) * TILE_WIDTH + tx;
    int h = (bz / W_tile) * TILE_WIDTH + ty;


    // float acc = 0.0f;
    __half acc = __float2half(0.0f);
    for (int c = 0; c < C; ++c) {
        if(K == 3){
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+0)), __float2half(mask_4d(by, c, 0, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+1)), __float2half(mask_4d(by, c, 0, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+2)), __float2half(mask_4d(by, c, 0, 2))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+0)), __float2half(mask_4d(by, c, 1, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+1)), __float2half(mask_4d(by, c, 1, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+2)), __float2half(mask_4d(by, c, 1, 2))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+0)), __float2half(mask_4d(by, c, 2, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+1)), __float2half(mask_4d(by, c, 2, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+2)), __float2half(mask_4d(by, c, 2, 2))));    
        }
        if (K > 3) {
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+0)), __float2half(mask_4d(by, c, 0, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+1)), __float2half(mask_4d(by, c, 0, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+2)), __float2half(mask_4d(by, c, 0, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+3)), __float2half(mask_4d(by, c, 0, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+4)), __float2half(mask_4d(by, c, 0, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+5)), __float2half(mask_4d(by, c, 0, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+0, w*S+6)), __float2half(mask_4d(by, c, 0, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+0)), __float2half(mask_4d(by, c, 1, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+1)), __float2half(mask_4d(by, c, 1, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+2)), __float2half(mask_4d(by, c, 1, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+3)), __float2half(mask_4d(by, c, 1, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+4)), __float2half(mask_4d(by, c, 1, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+5)), __float2half(mask_4d(by, c, 1, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+1, w*S+6)), __float2half(mask_4d(by, c, 1, 6))));


            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+0)), __float2half(mask_4d(by, c, 2, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+1)), __float2half(mask_4d(by, c, 2, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+2)), __float2half(mask_4d(by, c, 2, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+3)), __float2half(mask_4d(by, c, 2, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+4)), __float2half(mask_4d(by, c, 2, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+5)), __float2half(mask_4d(by, c, 2, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+2, w*S+6)), __float2half(mask_4d(by, c, 2, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+0)), __float2half(mask_4d(by, c, 3, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+1)), __float2half(mask_4d(by, c, 3, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+2)), __float2half(mask_4d(by, c, 3, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+3)), __float2half(mask_4d(by, c, 3, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+4)), __float2half(mask_4d(by, c, 3, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+5)), __float2half(mask_4d(by, c, 3, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+3, w*S+6)), __float2half(mask_4d(by, c, 3, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+0)), __float2half(mask_4d(by, c, 4, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+1)), __float2half(mask_4d(by, c, 4, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+2)), __float2half(mask_4d(by, c, 4, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+3)), __float2half(mask_4d(by, c, 4, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+4)), __float2half(mask_4d(by, c, 4, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+5)), __float2half(mask_4d(by, c, 4, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+4, w*S+6)), __float2half(mask_4d(by, c, 4, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+0)), __float2half(mask_4d(by, c, 5, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+1)), __float2half(mask_4d(by, c, 5, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+2)), __float2half(mask_4d(by, c, 5, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+3)), __float2half(mask_4d(by, c, 5, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+4)), __float2half(mask_4d(by, c, 5, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+5)), __float2half(mask_4d(by, c, 5, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+5, w*S+6)), __float2half(mask_4d(by, c, 5, 6))));

            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+0)), __float2half(mask_4d(by, c, 6, 0))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+1)), __float2half(mask_4d(by, c, 6, 1))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+2)), __float2half(mask_4d(by, c, 6, 2))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+3)), __float2half(mask_4d(by, c, 6, 3))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+4)), __float2half(mask_4d(by, c, 6, 4))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+5)), __float2half(mask_4d(by, c, 6, 5))));
            acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+6, w*S+6)), __float2half(mask_4d(by, c, 6, 6))));
        } 
    }
    if(h < H_out && w < W_out){
        atomicAdd(&out_4d(bx, by, h, w),acc);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_atomicadd(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */
    
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // using constant memory

    // Insert your GPU convolution kernel code here
    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int h = (blockIdx.z / W_tile) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_tile) * TILE_WIDTH + threadIdx.x;
    // float acc = 0.0f;
    __half acc = __float2half(0.0f);


    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if (h * S + p < H && w * S + q < W) {
                    // acc += in_4d(bx, c, h*S+p, w*S+q) * mask_4d(by, c, p, q);
                    acc = __hadd(acc, __hmul(__float2half (in_4d(bx, c, h*S+p, w*S+q)), __float2half(mask_4d(by, c, p, q))));
                }
            }
        }
    }
    if (h < H_out && w < W_out) {
        atomicAdd(&out_4d(bx, by, h, w),acc);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__global__ void conv_forward_kernel_tiling(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S){
   
    extern __shared__ float subtile[];
    
    const int H_out = (H - K) / S + 1; // Output height dimension
    const int W_out = (W - K) / S + 1; // Output width dimension
    int t_size = (TILE_WIDTH - 1) * S + K;

    int W_tile = ceil(1.0 * W_out / TILE_WIDTH); // W_size
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int h = (blockIdx.z / W_tile) * TILE_WIDTH + ty;
    int w = (blockIdx.z % W_tile) * TILE_WIDTH + tx;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) CM[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define st(i2, i1, i0) subtile[(i2) * (t_size * t_size) + (i1) * (t_size) + i0]

    // pre-load the input array into shared memory
    for (int c = 0; c < C; c++)
        for (int p = ty; p < t_size; p += TILE_WIDTH)
            for (int q = tx; q < t_size; q += TILE_WIDTH)
                st(c, p, q) = in_4d(b_x, c, (h - ty) * S + p, (w - tx) * S + q);

    __syncthreads();
    if (h < H_out && w < W_out) {
        float acc = 0.0;
        for (int c = 0; c < C; c++)
            for (int p = 0; p < K; p++)
                for (int q = 0; q < K; q++)
                    if(ty * S + p < t_size && tx * S + q < t_size)
                        acc += st(c, ty * S + p, tx * S + q) * mask_4d(b_y, c, p, q);
        out_4d(b_x, b_y, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
}

// Tiled
__global__ void conv_forward_kernel_tiled(float *output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{  
    extern __shared__ float tile_input[];
    
    const int H_out = (H - K) / S + 1; // Output height dimension
    const int W_out = (W - K) / S + 1; // Output width dimension
    int t_size = (TILE_WIDTH - 1) * S + K;

    int W_tile = ceil(1.0 * W_out / TILE_WIDTH); // W_size
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bxi = blockIdx.x;
    int h = (blockIdx.y / W_tile) * TILE_WIDTH + ty;
    int w = (blockIdx.y % W_tile) * TILE_WIDTH + tx;
    int bzi = blockIdx.z;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    //#define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d(i3, i2, i1, i0) CM [(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // using constant memory
    #define tile_3d(i2, i1, i0) tile_input[(i2) * (t_size * t_size) + (i1) * (t_size) + i0]

    // pre-load the input array into shared memory
    for (int c = 0; c < C; c++)
        for (int p = ty; p < t_size; p += TILE_WIDTH)
            for (int q = tx; q < t_size; q += TILE_WIDTH)
                tile_3d(c, p, q) = in_4d(bzi, c, (h - ty) * S + p, (w - tx) * S + q);

    __syncthreads();
    if (h < H_out && w < W_out) {
        // __half acc = __float2half(0.0f);
        float acc = 0.0;
        for (int c = 0; c < C; c++){
            // for (int p = 0; p < K; p++)
            //     for (int q = 0; q < K; q++)
            // if(ty * S + p < t_size && tx * S + q < t_size)
                // acc += tile_3d(c, ty * S + p, tx * S + q) * mask_4d(bxi, c, p, q);
                if(K == 3){
                    acc += tile_3d(c, ty*S+0, tx*S+0) * mask_4d(bxi, c, 0, 0);
                    acc += tile_3d(c, ty*S+0, tx*S+1) * mask_4d(bxi, c, 0, 1);
                    acc += tile_3d(c, ty*S+0, tx*S+2) * mask_4d(bxi, c, 0, 2);

                    acc += tile_3d(c, ty*S+1, tx*S+0) * mask_4d(bxi, c, 1, 0);
                    acc += tile_3d(c, ty*S+1, tx*S+1) * mask_4d(bxi, c, 1, 1);
                    acc += tile_3d(c, ty*S+1, tx*S+2) * mask_4d(bxi, c, 1, 2);

                    acc += tile_3d(c, ty*S+2, tx*S+0) * mask_4d(bxi, c, 2, 0);
                    acc += tile_3d(c, ty*S+2, tx*S+1) * mask_4d(bxi, c, 2, 1);
                    acc += tile_3d(c, ty*S+2, tx*S+2) * mask_4d(bxi, c, 2, 2);
                }
                if (K > 3) {
                    acc += tile_3d(c, ty*S+0, tx*S+0) * mask_4d(bxi, c, 0, 0);
                    acc += tile_3d(c, ty*S+0, tx*S+1) * mask_4d(bxi, c, 0, 1);
                    acc += tile_3d(c, ty*S+0, tx*S+2) * mask_4d(bxi, c, 0, 2);                
                    acc += tile_3d(c, ty*S+0, tx*S+3) * mask_4d(bxi, c, 0, 3);
                    acc += tile_3d(c, ty*S+0, tx*S+4) * mask_4d(bxi, c, 0, 4);
                    acc += tile_3d(c, ty*S+0, tx*S+5) * mask_4d(bxi, c, 0, 5);
                    acc += tile_3d(c, ty*S+0, tx*S+6) * mask_4d(bxi, c, 0, 6);
                
                    acc += tile_3d(c, ty*S+1, tx*S+0) * mask_4d(bxi, c, 1, 0);
                    acc += tile_3d(c, ty*S+1, tx*S+1) * mask_4d(bxi, c, 1, 1);
                    acc += tile_3d(c, ty*S+1, tx*S+2) * mask_4d(bxi, c, 1, 2);
                    acc += tile_3d(c, ty*S+1, tx*S+3) * mask_4d(bxi, c, 1, 3);
                    acc += tile_3d(c, ty*S+1, tx*S+4) * mask_4d(bxi, c, 1, 4);
                    acc += tile_3d(c, ty*S+1, tx*S+5) * mask_4d(bxi, c, 1, 5);
                    acc += tile_3d(c, ty*S+1, tx*S+6) * mask_4d(bxi, c, 1, 6);
                
                    acc += tile_3d(c, ty*S+2, tx*S+0) * mask_4d(bxi, c, 2, 0);
                    acc += tile_3d(c, ty*S+2, tx*S+1) * mask_4d(bxi, c, 2, 1);
                    acc += tile_3d(c, ty*S+2, tx*S+2) * mask_4d(bxi, c, 2, 2);
                    acc += tile_3d(c, ty*S+2, tx*S+3) * mask_4d(bxi, c, 2, 3);
                    acc += tile_3d(c, ty*S+2, tx*S+4) * mask_4d(bxi, c, 2, 4);
                    acc += tile_3d(c, ty*S+2, tx*S+5) * mask_4d(bxi, c, 2, 5);
                    acc += tile_3d(c, ty*S+2, tx*S+6) * mask_4d(bxi, c, 2, 6);
                
                    acc += tile_3d(c, ty*S+3, tx*S+0) * mask_4d(bxi, c, 3, 0);
                    acc += tile_3d(c, ty*S+3, tx*S+1) * mask_4d(bxi, c, 3, 1);
                    acc += tile_3d(c, ty*S+3, tx*S+2) * mask_4d(bxi, c, 3, 2);
                    acc += tile_3d(c, ty*S+3, tx*S+3) * mask_4d(bxi, c, 3, 3);
                    acc += tile_3d(c, ty*S+3, tx*S+4) * mask_4d(bxi, c, 3, 4);
                    acc += tile_3d(c, ty*S+3, tx*S+5) * mask_4d(bxi, c, 3, 5);
                    acc += tile_3d(c, ty*S+3, tx*S+6) * mask_4d(bxi, c, 3, 6);
                
                    acc += tile_3d(c, ty*S+4, tx*S+0) * mask_4d(bxi, c, 4, 0);
                    acc += tile_3d(c, ty*S+4, tx*S+1) * mask_4d(bxi, c, 4, 1);
                    acc += tile_3d(c, ty*S+4, tx*S+2) * mask_4d(bxi, c, 4, 2);
                    acc += tile_3d(c, ty*S+4, tx*S+3) * mask_4d(bxi, c, 4, 3);
                    acc += tile_3d(c, ty*S+4, tx*S+4) * mask_4d(bxi, c, 4, 4);
                    acc += tile_3d(c, ty*S+4, tx*S+5) * mask_4d(bxi, c, 4, 5);
                    acc += tile_3d(c, ty*S+4, tx*S+6) * mask_4d(bxi, c, 4, 6);
                
                    acc += tile_3d(c, ty*S+5, tx*S+0) * mask_4d(bxi, c, 5, 0);
                    acc += tile_3d(c, ty*S+5, tx*S+1) * mask_4d(bxi, c, 5, 1);
                    acc += tile_3d(c, ty*S+5, tx*S+2) * mask_4d(bxi, c, 5, 2);
                    acc += tile_3d(c, ty*S+5, tx*S+3) * mask_4d(bxi, c, 5, 3);
                    acc += tile_3d(c, ty*S+5, tx*S+4) * mask_4d(bxi, c, 5, 4);
                    acc += tile_3d(c, ty*S+5, tx*S+5) * mask_4d(bxi, c, 5, 5);
                    acc += tile_3d(c, ty*S+5, tx*S+6) * mask_4d(bxi, c, 5, 6);
                
                    acc += tile_3d(c, ty*S+6, tx*S+0) * mask_4d(bxi, c, 6, 0);
                    acc += tile_3d(c, ty*S+6, tx*S+1) * mask_4d(bxi, c, 6, 1);
                    acc += tile_3d(c, ty*S+6, tx*S+2) * mask_4d(bxi, c, 6, 2);
                    acc += tile_3d(c, ty*S+6, tx*S+3) * mask_4d(bxi, c, 6, 3);
                    acc += tile_3d(c, ty*S+6, tx*S+4) * mask_4d(bxi, c, 6, 4);
                    acc += tile_3d(c, ty*S+6, tx*S+5) * mask_4d(bxi, c, 6, 5);
                    acc += tile_3d(c, ty*S+6, tx*S+6) * mask_4d(bxi, c, 6, 6);
                }
        }
        out_4d(bzi, bxi, h, w) = acc;
        // atomicAdd(&out_4d(bzi, bxi, h, w), acc);
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef tile_3d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory for device_output, device_input, and device_mask
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    cudaMalloc((void**) device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void**) device_input_ptr, B * C * H * W * sizeof(float));
    // cudaMalloc((void**) device_mask_ptr, M * C * K * K * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(CM, host_mask, M * C * K * K * sizeof(float));
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    int W_tile = ceil(1.0 * W_out / TILE_WIDTH);
    int H_tile = ceil(1.0 * H_out / TILE_WIDTH);

    int Z = W_tile * H_tile;
    int t_size = (TILE_WIDTH - 1) * S + K;
    size_t share = (C * t_size * t_size * sizeof(float));

    // dim3 dimgrid(B, M, Z); // Original
    dim3 dimgrid_tiled(M, Z, B); // tiled

    dim3 dimblock(TILE_WIDTH, TILE_WIDTH, 1);
    // conv_forward_kernel <<<dimgrid, dimblock, 1>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
    // conv_forward_kernel_constant_mem <<<dimgrid, dimblock, 1>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
    // conv_forward_kernel_unroll <<<dimgrid, dimblock, 1>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
    // conv_forward_kernel_fp16 <<<dimgrid, dimblock, 1>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
    // conv_forward_kernel_unroll_fp16 <<<dimgrid, dimblock, 1>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
    // conv_forward_kernel_unroll_atomicadd <<<dimgrid, dimblock, 1>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);
    conv_forward_kernel_tiled <<<dimgrid_tiled, dimblock, share>>> (device_output, device_input, device_mask, B, M, C, H, W, K, S);

}



__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, B * M * W_out * H_out * sizeof(float), cudaMemcpyDeviceToHost);
   
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
