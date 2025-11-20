#include "grayscale.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        
        if (channels >= 3) {
            unsigned char r = input[idx];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];
            
            // 0.299*R + 0.587*G + 0.114*B
            unsigned char gray = (77 * r + 150 * g + 29 * b) / 256;
            output[y * width + x] = gray;
        } else if (channels == 1) {
            output[y * width + x] = input[idx];
        }
    }
}

__global__ void grayscaleWeightedKernel(const unsigned char* input, unsigned char* output,
                                        int width, int height, int channels,
                                        float r_weight, float g_weight, float b_weight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        
        if (channels >= 3) {
            unsigned char r = input[idx];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];
            
            float gray_f = r_weight * r + g_weight * g + b_weight * b;
            unsigned char gray = (unsigned char)fminf(fmaxf(gray_f, 0.0f), 255.0f);
            
            output[y * width + x] = gray;
        } else if (channels == 1) {
            output[y * width + x] = input[idx];
        }
    }
}

__global__ void grayscaleInPlaceKernel(unsigned char* data, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height && channels >= 3) {
        int idx = (y * width + x) * channels;
        
        unsigned char r = data[idx];
        unsigned char g = data[idx + 1];
        unsigned char b = data[idx + 2];
        
        unsigned char gray = (77 * r + 150 * g + 29 * b) / 256;
        
        data[idx] = gray;
        data[idx + 1] = gray;
        data[idx + 2] = gray;
    }
}

extern "C" void launchGrayscaleKernel(const unsigned char* input, unsigned char* output,
                                      int width, int height, int channels, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    grayscaleKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height, channels);
    if (stream == 0) {
        cudaDeviceSynchronize();
    }
}

extern "C" void launchGrayscaleWeightedKernel(const unsigned char* input, unsigned char* output,
                                              int width, int height, int channels,
                                              float r_weight, float g_weight, float b_weight,
                                              cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    grayscaleWeightedKernel<<<gridSize, blockSize, 0, stream>>>(input, output, width, height, channels,
                                                      r_weight, g_weight, b_weight);
    if (stream == 0) {
        cudaDeviceSynchronize();
    }
}

bool GrayscaleFilter::apply(const ImageData& input, ImageData& output) {
    if (!input.gpu_data || input.channels < 1) {
        fprintf(stderr, "Invalid input image for grayscale filter\n");
        return false;
    }
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = 1;
        output.size_bytes = input.width * input.height * sizeof(unsigned char);
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    grayscaleKernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                             input.width, input.height, input.channels);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool GrayscaleFilter::applyInPlace(ImageData& image) {
    if (!image.gpu_data || image.channels < 3) {
        fprintf(stderr, "Invalid image for in-place grayscale filter\n");
        return false;
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((image.width + blockSize.x - 1) / blockSize.x,
                  (image.height + blockSize.y - 1) / blockSize.y);
    
    grayscaleInPlaceKernel<<<gridSize, blockSize>>>(image.gpu_data, image.width,
                                                     image.height, image.channels);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

bool GrayscaleFilter::applyWithWeights(const ImageData& input, ImageData& output,
                                       float r_weight, float g_weight, float b_weight) {
    if (!input.gpu_data || input.channels < 3) {
        fprintf(stderr, "Invalid input image for weighted grayscale filter\n");
        return false;
    }
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = 1;
        output.size_bytes = input.width * input.height * sizeof(unsigned char);
        output.data = new unsigned char[output.size_bytes];
    }
    
    if (!output.gpu_data) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&output.gpu_data, output.size_bytes));
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    grayscaleWeightedKernel<<<gridSize, blockSize>>>(input.gpu_data, output.gpu_data,
                                                      input.width, input.height, input.channels,
                                                      r_weight, g_weight, b_weight);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    return true;
}

// ============================================================================
// Версии с поддержкой CUDA stream (для конвейерной обработки)
// ============================================================================

bool GrayscaleFilter::apply(ImageData& image, cudaStream_t stream) {
    if (!image.gpu_data || image.channels < 1) {
        fprintf(stderr, "Invalid input image for grayscale filter\n");
        return false;
    }
    
    // Создаем временный буфер для результата (1 канал)
    unsigned char* output_gpu = nullptr;
    size_t output_size = image.width * image.height * sizeof(unsigned char);
    CUDA_CHECK_RETURN(cudaMalloc((void**)&output_gpu, output_size));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((image.width + blockSize.x - 1) / blockSize.x,
                  (image.height + blockSize.y - 1) / blockSize.y);
    
    grayscaleKernel<<<gridSize, blockSize, 0, stream>>>(image.gpu_data, output_gpu,
                                                        image.width, image.height, image.channels);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    
    // Освобождаем старый буфер и заменяем на новый
    if (image.gpu_data) {
        cudaFree(image.gpu_data);
    }
    image.gpu_data = output_gpu;
    image.channels = 1;
    image.size_bytes = output_size;
    
    // Обновляем host буфер
    if (image.data) {
        delete[] image.data;
    }
    image.data = new unsigned char[output_size];
    
    return true;
}

bool GrayscaleFilter::applyWithWeights(ImageData& image, cudaStream_t stream,
                                       float r_weight, float g_weight, float b_weight) {
    if (!image.gpu_data || image.channels < 3) {
        fprintf(stderr, "Invalid input image for weighted grayscale filter\n");
        return false;
    }
    
    // Создаем временный буфер для результата (1 канал)
    unsigned char* output_gpu = nullptr;
    size_t output_size = image.width * image.height * sizeof(unsigned char);
    CUDA_CHECK_RETURN(cudaMalloc((void**)&output_gpu, output_size));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((image.width + blockSize.x - 1) / blockSize.x,
                  (image.height + blockSize.y - 1) / blockSize.y);
    
    grayscaleWeightedKernel<<<gridSize, blockSize, 0, stream>>>(image.gpu_data, output_gpu,
                                                                image.width, image.height, image.channels,
                                                                r_weight, g_weight, b_weight);
    
    CUDA_CHECK_RETURN(cudaGetLastError());
    
    // Освобождаем старый буфер и заменяем на новый
    if (image.gpu_data) {
        cudaFree(image.gpu_data);
    }
    image.gpu_data = output_gpu;
    image.channels = 1;
    image.size_bytes = output_size;
    
    // Обновляем host буфер
    if (image.data) {
        delete[] image.data;
    }
    image.data = new unsigned char[output_size];
    
    return true;
}


