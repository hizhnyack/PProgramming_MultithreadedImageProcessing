#include "image_processor.h"
#include <iostream>

ImageProcessor::ImageProcessor() : device_count_(0), initialized_(false) {
    initializeCuda();
}

ImageProcessor::~ImageProcessor() {
    cleanup();
}

void ImageProcessor::initializeCuda() {
    cudaError_t error = cudaGetDeviceCount(&device_count_);
    if (error != cudaSuccess || device_count_ == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        throw std::runtime_error("CUDA initialization failed");
    }
    initialized_ = true;
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
