#ifndef BLUR_H
#define BLUR_H

#include "../core/image_processor.h"
#include "../core/cuda_utils.h"

// Размытие изображения
class BlurFilter {
public:
    static bool applyBox(const ImageData& input, ImageData& output, int radius);
    static bool applyGaussian(const ImageData& input, ImageData& output, float sigma);
    static bool applyGaussianSeparable(const ImageData& input, ImageData& output, float sigma);
    static bool applyMotion(const ImageData& input, ImageData& output, 
                            int length, float angle);
};

// Низкоуровневые CUDA функции
extern "C" {
    void launchBoxBlurKernel(const unsigned char* input, unsigned char* output,
                             int width, int height, int channels, int radius);
    
    void launchGaussianBlurKernel(const unsigned char* input, unsigned char* output,
                                  int width, int height, int channels,
                                  const float* kernel, int kernel_size);
    
    void launchGaussianBlurSeparableKernel(const unsigned char* input, unsigned char* temp,
                                           unsigned char* output, int width, int height,
                                           int channels, const float* kernel, int kernel_size);
    
    void launchMotionBlurKernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels,
                                int length, float cos_angle, float sin_angle);
}

#endif // BLUR_H

