#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "cuda_utils.h"
#include "image_data.h"

// ImageData теперь определен в image_data.h для избежания зависимостей от CUDA в CPU версиях

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();
    
    bool allocateGPUMemory(ImageData& img_data);
    void freeGPUMemory(ImageData& img_data);
    
    bool copyHostToDevice(const ImageData& host_img, ImageData& device_img);
    bool copyDeviceToHost(const ImageData& device_img, ImageData& host_img);
    
    size_t calculateImageSize(int width, int height, int channels) const;
    bool validateImage(const ImageData& img) const;
    
    int getDeviceCount() const;
    void setDevice(int device_id) const;
    void synchronizeDevice() const;
    
    cudaEvent_t createEvent();
    void recordEvent(cudaEvent_t event);
    float getElapsedTime(cudaEvent_t start, cudaEvent_t end);

private:
    int device_count_;
    bool initialized_;
    
    void initializeCuda();
    void cleanup();
};

#endif 
