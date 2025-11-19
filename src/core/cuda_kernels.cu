#include "cuda_runtime.h"
#include <stdio.h>

__global__ void validateImageKernel(unsigned char* data, int size, int* error_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        if (data[idx] == 0) {  
            atomicAdd(error_count, 1);
        }
    }
}

__global__ void clearMemoryKernel(unsigned char* data, int size, unsigned char value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = value;
    }
}

extern "C" int validateImageData(unsigned char* gpu_data, int size) {
    int* d_error_count;
    int h_error_count = 0;
    
    cudaMalloc((void**)&d_error_count, sizeof(int));
    cudaMemset(d_error_count, 0, sizeof(int));
    
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    validateImageKernel<<<blocks, threads_per_block>>>(gpu_data, size, d_error_count);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_error_count, d_error_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_error_count);
    
    return h_error_count;
}

extern "C" void clearGPUMemory(unsigned char* gpu_data, int size, unsigned char value) {
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    clearMemoryKernel<<<blocks, threads_per_block>>>(gpu_data, size, value);
    cudaDeviceSynchronize();
}
