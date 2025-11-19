#include "image_loader.h"
#include "../core/cuda_utils.h"
#include <stdio.h>
#include <string.h>

// STB Image - header-only библиотека для загрузки изображений
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

bool ImageLoader::load(const std::string& filename, ImageData& image) {
    // Освобождаем старые данные если есть
    if (image.data) {
        delete[] image.data;
        image.data = nullptr;
    }
    if (image.gpu_data) {
        cudaFree(image.gpu_data);
        image.gpu_data = nullptr;
    }
    
    // Загружаем изображение
    int width, height, channels;
    unsigned char* loaded_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    
    if (!loaded_data) {
        fprintf(stderr, "Failed to load image: %s\n", filename.c_str());
        return false;
    }
    
    // Заполняем структуру
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.size_bytes = width * height * channels;
    
    // Копируем данные на host
    image.data = new unsigned char[image.size_bytes];
    memcpy(image.data, loaded_data, image.size_bytes);
    
    // Освобождаем временный буфер STB
    stbi_image_free(loaded_data);
    
    // Выделяем память на GPU
    CUDA_CHECK_RETURN(cudaMalloc(&image.gpu_data, image.size_bytes));
    CUDA_CHECK_RETURN(cudaMemcpy(image.gpu_data, image.data, 
                                  image.size_bytes, cudaMemcpyHostToDevice));
    
    printf("Loaded image: %s (%dx%d, %d channels)\n", 
           filename.c_str(), width, height, channels);
    
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

