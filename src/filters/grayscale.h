#ifndef GRAYSCALE_H
#define GRAYSCALE_H

#include "../core/image_processor.h"
#include "../core/cuda_utils.h"

// Преобразование изображения в оттенки серого
class GrayscaleFilter {
public:
    // Версии без stream (по умолчанию используют stream 0)
    static bool apply(const ImageData& input, ImageData& output);
    static bool applyInPlace(ImageData& image);
    static bool applyWithWeights(const ImageData& input, ImageData& output,
                                 float r_weight, float g_weight, float b_weight);
    
    // Версии с поддержкой CUDA stream (для конвейерной обработки)
    static bool apply(ImageData& image, cudaStream_t stream);
    static bool applyWithWeights(ImageData& image, cudaStream_t stream,
                                 float r_weight, float g_weight, float b_weight);
};

// Низкоуровневые CUDA функции
extern "C" {
    void launchGrayscaleKernel(const unsigned char* input, unsigned char* output,
                               int width, int height, int channels, cudaStream_t stream = 0);
    
    void launchGrayscaleWeightedKernel(const unsigned char* input, unsigned char* output,
                                       int width, int height, int channels,
                                       float r_weight, float g_weight, float b_weight,
                                       cudaStream_t stream = 0);
}

#endif // GRAYSCALE_H


