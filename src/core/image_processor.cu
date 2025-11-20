#include "image_processor.h"
#include <iostream>
#include <unistd.h>  // для sleep()

ImageProcessor::ImageProcessor() : device_count_(0), initialized_(false) {
    initializeCuda();
}

ImageProcessor::~ImageProcessor() {
    cleanup();
}

void ImageProcessor::initializeCuda() {
    fprintf(stderr, "[LOG] ImageProcessor::initializeCuda() called\n");
    
    cudaError_t error;
    int max_retries = 3;
    int retry_delay = 2;  // секунды
    
    // Проверяем переменные окружения
    const char* cuda_visible = getenv("CUDA_VISIBLE_DEVICES");
    fprintf(stderr, "[LOG] CUDA_VISIBLE_DEVICES=%s\n", cuda_visible ? cuda_visible : "(not set)");
    
    // Пытаемся найти CUDA устройство с повторными попытками
    for (int attempt = 0; attempt < max_retries; attempt++) {
        fprintf(stderr, "[LOG] Attempt %d/%d: Checking for CUDA devices...\n", 
                attempt + 1, max_retries);
        
        error = cudaGetDeviceCount(&device_count_);
        fprintf(stderr, "[LOG] cudaGetDeviceCount() returned: error=%s, count=%d\n", 
                cudaGetErrorString(error), device_count_);
        
        if (error == cudaSuccess && device_count_ > 0) {
            fprintf(stderr, "[LOG] CUDA device found!\n");
            break;  // Устройство найдено
        }
        
        if (attempt < max_retries - 1) {
            fprintf(stderr, "[WARN] CUDA device not found (attempt %d/%d), waiting %d seconds...\n", 
                    attempt + 1, max_retries, retry_delay);
            sleep(retry_delay);
        }
    }
    
    if (error != cudaSuccess || device_count_ == 0) {
        fprintf(stderr, "[ERROR] No CUDA-capable devices found after %d attempts\n", max_retries);
        fprintf(stderr, "[ERROR] CUDA error: %s\n", cudaGetErrorString(error));
        fprintf(stderr, "[ERROR] Device count: %d\n", device_count_);
        throw std::runtime_error("CUDA initialization failed");
    }
    
    // Явно выбираем первое устройство (NVIDIA GPU)
    fprintf(stderr, "[LOG] Setting CUDA device to 0...\n");
    error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        fprintf(stderr, "[ERROR] Failed to set CUDA device 0: %s\n", cudaGetErrorString(error));
        throw std::runtime_error("CUDA device selection failed");
    }
    fprintf(stderr, "[LOG] CUDA device 0 set successfully\n");
    
    // Проверяем текущее устройство
    int current_device = -1;
    cudaGetDevice(&current_device);
    fprintf(stderr, "[LOG] Current CUDA device: %d\n", current_device);
    
    // Получаем и выводим информацию об устройстве
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error == cudaSuccess) {
        fprintf(stderr, "[LOG] Using CUDA device: %s (Compute %d.%d, %zu MB memory)\n", 
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    } else {
        fprintf(stderr, "[WARN] Failed to get device properties: %s\n", cudaGetErrorString(error));
    }
    
    initialized_ = true;
    fprintf(stderr, "[LOG] CUDA initialization completed successfully\n");
}

void ImageProcessor::cleanup() {
    // Cleanup is handled by ImageData destructors
    // Additional cleanup if needed
    cudaDeviceReset();
}

bool ImageProcessor::allocateGPUMemory(ImageData& img_data) {
    if (!validateImage(img_data)) {
        return false;
    }
    
    if (img_data.gpu_data) {
        cudaFree(img_data.gpu_data);
        img_data.gpu_data = nullptr;
    }
    
    cudaError_t error = cudaMalloc((void**)&img_data.gpu_data, img_data.size_bytes);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    return img_data.gpu_data != nullptr;
}

void ImageProcessor::freeGPUMemory(ImageData& img_data) {
    if (img_data.gpu_data) {
        cudaFree(img_data.gpu_data);
        img_data.gpu_data = nullptr;
    }
}

bool ImageProcessor::copyHostToDevice(const ImageData& host_img, ImageData& device_img) {
    if (!validateImage(host_img) || !validateImage(device_img)) {
        return false;
    }
    
    if (!device_img.gpu_data) {
        if (!allocateGPUMemory(device_img)) {
            return false;
        }
    }
    
    cudaError_t error = cudaMemcpy(device_img.gpu_data, host_img.data, 
                                  host_img.size_bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to GPU: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    return true;
}

bool ImageProcessor::copyDeviceToHost(const ImageData& device_img, ImageData& host_img) {
    if (!validateImage(device_img) || !validateImage(host_img)) {
        return false;
    }
    
    cudaError_t error = cudaMemcpy(host_img.data, device_img.gpu_data, 
                                  device_img.size_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from GPU: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    return true;
}

size_t ImageProcessor::calculateImageSize(int width, int height, int channels) const {
    return static_cast<size_t>(width) * height * channels * sizeof(unsigned char);
}

bool ImageProcessor::validateImage(const ImageData& img) const {
    return img.data != nullptr && 
           img.width > 0 && 
           img.height > 0 && 
           img.channels > 0 && 
           img.size_bytes > 0;
}

int ImageProcessor::getDeviceCount() const {
    return device_count_;
}

void ImageProcessor::setDevice(int device_id) const {
    if (device_id >= 0 && device_id < device_count_) {
        CUDA_CHECK(cudaSetDevice(device_id));
    }
}

void ImageProcessor::synchronizeDevice() const {
    CUDA_CHECK(cudaDeviceSynchronize());
}

cudaEvent_t ImageProcessor::createEvent() {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    return event;
}

void ImageProcessor::recordEvent(cudaEvent_t event) {
    CUDA_CHECK(cudaEventRecord(event));
}

float ImageProcessor::getElapsedTime(cudaEvent_t start, cudaEvent_t end) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    return milliseconds;
}

// Реализация cleanupGPU для ImageData
void ImageData::cleanupGPU() {
    if (gpu_data) {
        cudaFree(gpu_data);
        gpu_data = nullptr;
    }
}
