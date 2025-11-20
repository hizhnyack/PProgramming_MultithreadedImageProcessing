#include "image_loader.h"
#include "../core/cuda_utils.h"
#include <stdio.h>
#include <string.h>

// STB Image - header-only библиотека для загрузки изображений
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

bool ImageLoader::load(const std::string& filename, ImageData& image, bool use_gpu) {
    fprintf(stderr, "[LOG] ImageLoader::load() called for file: %s (use_gpu=%s)\n", 
            filename.c_str(), use_gpu ? "true" : "false");
    
    // Освобождаем старые данные если есть
    if (image.data) {
        delete[] image.data;
        image.data = nullptr;
    }
    if (image.gpu_data) {
        fprintf(stderr, "[LOG] Freeing old GPU memory\n");
        if (use_gpu) {
            cudaFree(image.gpu_data);
        }
        image.gpu_data = nullptr;
    }
    
    // Загружаем изображение
    fprintf(stderr, "[LOG] Loading image from disk...\n");
    int width, height, channels;
    unsigned char* loaded_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    
    if (!loaded_data) {
        fprintf(stderr, "[ERROR] Failed to load image: %s\n", filename.c_str());
        return false;
    }
    
    fprintf(stderr, "[LOG] Image loaded: %dx%d, %d channels\n", width, height, channels);
    
    // Заполняем структуру
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.size_bytes = width * height * channels;
    
    fprintf(stderr, "[LOG] Image size: %zu bytes\n", image.size_bytes);
    
    // Копируем данные на host
    image.data = new unsigned char[image.size_bytes];
    memcpy(image.data, loaded_data, image.size_bytes);
    
    // Освобождаем временный буфер STB
    stbi_image_free(loaded_data);
    
    // Копируем на GPU только если нужно
    if (use_gpu) {
        // Проверяем доступность CUDA устройства перед использованием
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        fprintf(stderr, "[LOG] CUDA device count check: error=%s, count=%d\n", 
                cudaGetErrorString(error), device_count);
        
        if (error != cudaSuccess || device_count == 0) {
            fprintf(stderr, "[ERROR] No CUDA devices available before loading image!\n");
            fprintf(stderr, "[ERROR] CUDA error: %s\n", cudaGetErrorString(error));
            return false;
        }
        
        // Проверяем текущее устройство
        int current_device = -1;
        cudaGetDevice(&current_device);
        fprintf(stderr, "[LOG] Current CUDA device: %d\n", current_device);
        
        // Проверяем CUDA устройство еще раз перед выделением памяти
        error = cudaGetDeviceCount(&device_count);
        fprintf(stderr, "[LOG] Before cudaMalloc: device_count=%d, error=%s\n", 
                device_count, cudaGetErrorString(error));
        
        if (error != cudaSuccess || device_count == 0) {
            fprintf(stderr, "[ERROR] CUDA device became unavailable before memory allocation!\n");
            return false;
        }
        
        // Выделяем память на GPU
        fprintf(stderr, "[LOG] Allocating GPU memory: %zu bytes\n", image.size_bytes);
        error = cudaMalloc(&image.gpu_data, image.size_bytes);
        if (error != cudaSuccess) {
            fprintf(stderr, "[ERROR] cudaMalloc failed: %s\n", cudaGetErrorString(error));
            fprintf(stderr, "[ERROR] Device count: %d\n", device_count);
            return false;
        }
        fprintf(stderr, "[LOG] GPU memory allocated successfully at: %p\n", image.gpu_data);
        
        // Копируем данные на GPU
        fprintf(stderr, "[LOG] Copying data to GPU...\n");
        error = cudaMemcpy(image.gpu_data, image.data, 
                          image.size_bytes, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            fprintf(stderr, "[ERROR] cudaMemcpy failed: %s\n", cudaGetErrorString(error));
            cudaFree(image.gpu_data);
            image.gpu_data = nullptr;
            return false;
        }
        fprintf(stderr, "[LOG] Data copied to GPU successfully\n");
    } else {
        fprintf(stderr, "[LOG] Skipping GPU memory allocation (CPU mode)\n");
        image.gpu_data = nullptr;
    }
    
    printf("Loaded image: %s (%dx%d, %d channels) [%s]\n", 
           filename.c_str(), width, height, channels, use_gpu ? "GPU" : "CPU");
    
    return true;
}

bool ImageLoader::save(const std::string& filename, const ImageData& image) {
    if (!image.data) {
        fprintf(stderr, "No host data to save\n");
        return false;
    }
    
    // Определяем формат по расширению
    const char* ext = strrchr(filename.c_str(), '.');
    if (!ext) {
        fprintf(stderr, "No file extension found\n");
        return false;
    }
    
    int result = 0;
    
    if (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0) {
        result = stbi_write_png(filename.c_str(), image.width, image.height, 
                                image.channels, image.data, 
                                image.width * image.channels);
    } 
    else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 || 
             strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0) {
        result = stbi_write_jpg(filename.c_str(), image.width, image.height, 
                                image.channels, image.data, 90);
    }
    else if (strcmp(ext, ".bmp") == 0 || strcmp(ext, ".BMP") == 0) {
        result = stbi_write_bmp(filename.c_str(), image.width, image.height, 
                                image.channels, image.data);
    }
    else {
        fprintf(stderr, "Unsupported format: %s (use .png, .jpg, or .bmp)\n", ext);
        return false;
    }
    
    if (!result) {
        fprintf(stderr, "Failed to save image: %s\n", filename.c_str());
        return false;
    }
    
    printf("Saved image: %s\n", filename.c_str());
    return true;
}

bool ImageLoader::getInfo(const std::string& filename, int& width, int& height, int& channels) {
    int result = stbi_info(filename.c_str(), &width, &height, &channels);
    
    if (!result) {
        fprintf(stderr, "Failed to get image info: %s\n", filename.c_str());
        return false;
    }
    
    return true;
}

