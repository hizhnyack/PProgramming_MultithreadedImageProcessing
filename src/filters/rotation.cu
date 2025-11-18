#include "rotation.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void rotate90Kernel(const unsigned char* input, unsigned char* output,
                               int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int new_x = height - 1 - y;
        int new_y = x;
        
        int in_idx = (y * width + x) * channels;
        int out_idx = (new_y * height + new_x) * channels;
        
        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = input[in_idx + c];
        }
    }
}

__global__ void rotate180Kernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int new_x = width - 1 - x;
        int new_y = height - 1 - y;
        
        int in_idx = (y * width + x) * channels;
        int out_idx = (new_y * width + new_x) * channels;
        
        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = input[in_idx + c];
        }
    }
}

__global__ void rotate270Kernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int new_x = y;
        int new_y = width - 1 - x;
        
        int in_idx = (y * width + x) * channels;
        int out_idx = (new_y * height + new_x) * channels;
        
        for (int c = 0; c < channels; c++) {
            output[out_idx + c] = input[in_idx + c];
        }
    }
}

__global__ void rotateArbitraryKernel(const unsigned char* input, unsigned char* output,
                                      int width, int height, int channels,
                                      float cos_angle, float sin_angle,
                                      unsigned char background) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        float dx = x - cx;
        float dy = y - cy;
        
        float src_x = dx * cos_angle + dy * sin_angle + cx;
        float src_y = -dx * sin_angle + dy * cos_angle + cy;
        
        int out_idx = (y * width + x) * channels;
        
        if (src_x >= 0 && src_x < width - 1 && src_y >= 0 && src_y < height - 1) {
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float fx = src_x - x0;
            float fy = src_y - y0;
            
            for (int c = 0; c < channels; c++) {
                float v00 = input[(y0 * width + x0) * channels + c];
                float v10 = input[(y0 * width + x1) * channels + c];
                float v01 = input[(y1 * width + x0) * channels + c];
                float v11 = input[(y1 * width + x1) * channels + c];
                
                float v0 = v00 * (1 - fx) + v10 * fx;
                float v1 = v01 * (1 - fx) + v11 * fx;
                float v = v0 * (1 - fy) + v1 * fy;
                
                output[out_idx + c] = (unsigned char)v;
            }
        } else {
            for (int c = 0; c < channels; c++) {
                output[out_idx + c] = background;
            }
        }
    }
}

__global__ void rotate180InPlaceKernel(unsigned char* data, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int total_pixels = width * height;
    int idx = y * width + x;
    
    if (x < width && y < height && idx < total_pixels / 2) {
        int mirror_x = width - 1 - x;
        int mirror_y = height - 1 - y;
        
        int idx1 = (y * width + x) * channels;
        int idx2 = (mirror_y * width + mirror_x) * channels;
        
        for (int c = 0; c < channels; c++) {
            unsigned char temp = data[idx1 + c];
            data[idx1 + c] = data[idx2 + c];
            data[idx2 + c] = temp;
        }
    }
}

extern "C" void launchRotate90Kernel(const unsigned char* input, unsigned char* output,
                                     int width, int height, int channels) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    rotate90Kernel<<<gridSize, blockSize>>>(input, output, width, height, channels);
    cudaDeviceSynchronize();
}

extern "C" void launchRotate180Kernel(const unsigned char* input, unsigned char* output,
                                      int width, int height, int channels) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    rotate180Kernel<<<gridSize, blockSize>>>(input, output, width, height, channels);
    cudaDeviceSynchronize();
}

extern "C" void launchRotate270Kernel(const unsigned char* input, unsigned char* output,
                                      int width, int height, int channels) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    rotate270Kernel<<<gridSize, blockSize>>>(input, output, width, height, channels);
    cudaDeviceSynchronize();
}

extern "C" void launchRotateArbitraryKernel(const unsigned char* input, unsigned char* output,
                                            int width, int height, int channels,
                                            float angle_rad, unsigned char background) {
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    rotateArbitraryKernel<<<gridSize, blockSize>>>(input, output, width, height, channels,
                                                    cos_angle, sin_angle, background);
    cudaDeviceSynchronize();
}

bool RotationFilter::rotate90(const ImageData& input, ImageData& output) {
    if (!input.gpu_data || input.channels < 1) {
        fprintf(stderr, "Invalid input image for rotation filter\n");
        return false;
    }
    
    if (!output.data) {
        output.width = input.height;
        output.height = input.width;
        output.channels = input.channels;
        output.size_bytes = output.width * output.height * output.channels * sizeof(unsigned char);
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    rotate90Kernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                            input.width, input.height, input.channels);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool RotationFilter::rotate180(const ImageData& input, ImageData& output) {
    if (!input.gpu_data || input.channels < 1) {
        fprintf(stderr, "Invalid input image for rotation filter\n");
        return false;
    }
    
    // Подготовка выходного изображения (размеры остаются прежними)
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
    
    rotate180Kernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                             input.width, input.height, input.channels);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool RotationFilter::rotate270(const ImageData& input, ImageData& output) {
    if (!input.gpu_data || input.channels < 1) {
        fprintf(stderr, "Invalid input image for rotation filter\n");
        return false;
    }
    
    // Подготовка выходного изображения (размеры меняются местами)
    if (!output.data) {
        output.width = input.height;
        output.height = input.width;
        output.channels = input.channels;
        output.size_bytes = output.width * output.height * output.channels * sizeof(unsigned char);
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    rotate270Kernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                             input.width, input.height, input.channels);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool RotationFilter::rotateArbitrary(const ImageData& input, ImageData& output,
                                     float angle, unsigned char background) {
    if (!input.gpu_data || input.channels < 1) {
        fprintf(stderr, "Invalid input image for rotation filter\n");
        return false;
    }
    
    // Подготовка выходного изображения (размеры остаются прежними)
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
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);
    
    rotateArbitraryKernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                                    input.width, input.height, input.channels,
                                                    cos_angle, sin_angle, background);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool RotationFilter::rotate180InPlace(ImageData& image) {
    if (!image.gpu_data || image.channels < 1) {
        fprintf(stderr, "Invalid image for in-place rotation\n");
        return false;
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((image.width + blockSize.x - 1) / blockSize.x,
                  (image.height + blockSize.y - 1) / blockSize.y);
    
    rotate180InPlaceKernel<<<gridSize, blockSize>>>(image.gpu_data, image.width,
                                                     image.height, image.channels);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

