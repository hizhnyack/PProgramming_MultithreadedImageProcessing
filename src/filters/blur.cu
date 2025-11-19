#include "blur.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void boxBlurKernel(const unsigned char* input, unsigned char* output,
                              int width, int height, int channels, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            int count = 0;
            
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int idx = (ny * width + nx) * channels + c;
                        sum += input[idx];
                        count++;
                    }
                }
            }
            
            int out_idx = (y * width + x) * channels + c;
            output[out_idx] = (unsigned char)(sum / count);
        }
    }
}

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output,
                                   int width, int height, int channels,
                                   const float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int radius = kernel_size / 2;
        
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int nx = x + kx - radius;
                    int ny = y + ky - radius;
                    
                    nx = max(0, min(nx, width - 1));
                    ny = max(0, min(ny, height - 1));
                    
                    int idx = (ny * width + nx) * channels + c;
                    float weight = kernel[ky * kernel_size + kx];
                    sum += input[idx] * weight;
                }
            }
            
            int out_idx = (y * width + x) * channels + c;
            output[out_idx] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
        }
    }
}

__global__ void gaussianBlurHorizontalKernel(const unsigned char* input, unsigned char* output,
                                             int width, int height, int channels,
                                             const float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int radius = kernel_size / 2;
        
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int nx = x + k - radius;
                nx = max(0, min(nx, width - 1));
                
                int idx = (y * width + nx) * channels + c;
                sum += input[idx] * kernel[k];
            }
            
            int out_idx = (y * width + x) * channels + c;
            output[out_idx] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
        }
    }
}

__global__ void gaussianBlurVerticalKernel(const unsigned char* input, unsigned char* output,
                                           int width, int height, int channels,
                                           const float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int radius = kernel_size / 2;
        
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int ny = y + k - radius;
                ny = max(0, min(ny, height - 1));
                
                int idx = (ny * width + x) * channels + c;
                sum += input[idx] * kernel[k];
            }
            
            int out_idx = (y * width + x) * channels + c;
            output[out_idx] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
        }
    }
}

__global__ void motionBlurKernel(const unsigned char* input, unsigned char* output,
                                 int width, int height, int channels,
                                 int length, float cos_angle, float sin_angle) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            int count = 0;
            
            for (int i = -length / 2; i <= length / 2; i++) {
                int nx = x + (int)(i * cos_angle);
                int ny = y + (int)(i * sin_angle);
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int idx = (ny * width + nx) * channels + c;
                    sum += input[idx];
                    count++;
                }
            }
            
            int out_idx = (y * width + x) * channels + c;
            output[out_idx] = (unsigned char)(sum / count);
        }
    }
}

void generateGaussianKernel1D(float* kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        int x = i - radius;
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

void generateGaussianKernel2D(float* kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - radius;
            int dy = y - radius;
            float value = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            kernel[y * size + x] = value;
            sum += value;
        }
    }
    
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

extern "C" void launchBoxBlurKernel(const unsigned char* input, unsigned char* output,
                                    int width, int height, int channels, int radius) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    boxBlurKernel<<<gridSize, blockSize>>>(input, output, width, height, channels, radius);
    cudaDeviceSynchronize();
}

extern "C" void launchGaussianBlurKernel(const unsigned char* input, unsigned char* output,
                                         int width, int height, int channels,
                                         const float* kernel, int kernel_size) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    gaussianBlurKernel<<<gridSize, blockSize>>>(input, output, width, height, channels,
                                                kernel, kernel_size);
    cudaDeviceSynchronize();
}

extern "C" void launchGaussianBlurSeparableKernel(const unsigned char* input, unsigned char* temp,
                                                  unsigned char* output, int width, int height,
                                                  int channels, const float* kernel, int kernel_size) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    gaussianBlurHorizontalKernel<<<gridSize, blockSize>>>(input, temp, width, height, channels,
                                                          kernel, kernel_size);
    cudaDeviceSynchronize();
    
    gaussianBlurVerticalKernel<<<gridSize, blockSize>>>(temp, output, width, height, channels,
                                                        kernel, kernel_size);
    cudaDeviceSynchronize();
}

extern "C" void launchMotionBlurKernel(const unsigned char* input, unsigned char* output,
                                       int width, int height, int channels,
                                       int length, float cos_angle, float sin_angle) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    motionBlurKernel<<<gridSize, blockSize>>>(input, output, width, height, channels,
                                              length, cos_angle, sin_angle);
    cudaDeviceSynchronize();
}

bool BlurFilter::applyBox(const ImageData& input, ImageData& output, int radius) {
    if (!input.gpu_data || input.channels < 1 || radius < 1) {
        fprintf(stderr, "Invalid input for box blur filter\n");
        return false;
    }
    
    // Подготовка выходного изображения
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.size_bytes = input.size_bytes;
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    boxBlurKernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                           input.width, input.height, input.channels, radius);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool BlurFilter::applyGaussian(const ImageData& input, ImageData& output, float sigma) {
    if (!input.gpu_data || input.channels < 1 || sigma <= 0) {
        fprintf(stderr, "Invalid input for gaussian blur filter\n");
        return false;
    }
    
    int kernel_size = (int)(6 * sigma);
    if (kernel_size % 2 == 0) kernel_size++;
    if (kernel_size < 3) kernel_size = 3;
    if (kernel_size > 31) kernel_size = 31;
    
    float* h_kernel = new float[kernel_size * kernel_size];
    generateGaussianKernel2D(h_kernel, kernel_size, sigma);
    
    float* d_kernel;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float),
                                  cudaMemcpyHostToDevice));
    delete[] h_kernel;
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.size_bytes = input.size_bytes;
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    gaussianBlurKernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                                input.width, input.height, input.channels,
                                                d_kernel, kernel_size);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaFree(d_kernel);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool BlurFilter::applyGaussianSeparable(const ImageData& input, ImageData& output, float sigma) {
    if (!input.gpu_data || input.channels < 1 || sigma <= 0) {
        fprintf(stderr, "Invalid input for separable gaussian blur\n");
        return false;
    }
    
    int kernel_size = (int)(6 * sigma);
    if (kernel_size % 2 == 0) kernel_size++;
    if (kernel_size < 3) kernel_size = 3;
    if (kernel_size > 31) kernel_size = 31;
    
    float* h_kernel = new float[kernel_size];
    generateGaussianKernel1D(h_kernel, kernel_size, sigma);
    
    float* d_kernel;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), 
                                  cudaMemcpyHostToDevice));
    delete[] h_kernel;
    
    unsigned char* d_temp;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_temp, input.size_bytes));
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.size_bytes = input.size_bytes;
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    gaussianBlurHorizontalKernel<<<gridSize, blockSize>>>(input.gpu_data, d_temp,
                                                          input.width, input.height, input.channels,
                                                          d_kernel, kernel_size);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    
    gaussianBlurVerticalKernel<<<gridSize, blockSize>>>(d_temp, output.gpu_data,
                                                        input.width, input.height, input.channels,
                                                        d_kernel, kernel_size);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    cudaFree(d_kernel);
    cudaFree(d_temp);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool BlurFilter::applyMotion(const ImageData& input, ImageData& output,
                             int length, float angle) {
    if (!input.gpu_data || input.channels < 1 || length < 1) {
        fprintf(stderr, "Invalid input for motion blur\n");
        return false;
    }
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.size_bytes = input.size_bytes;
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    float angle_rad = angle * M_PI / 180.0f;
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    motionBlurKernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                              input.width, input.height, input.channels,
                                              length, cos_angle, sin_angle);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

