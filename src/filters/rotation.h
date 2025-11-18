#ifndef ROTATION_H
#define ROTATION_H

#include "../core/image_processor.h"
#include "../core/cuda_utils.h"

// Поворот изображения 90°, 180°, 270° и произвольный угол
class RotationFilter {
public:
    static bool rotate90(const ImageData& input, ImageData& output);
    static bool rotate180(const ImageData& input, ImageData& output);
    static bool rotate270(const ImageData& input, ImageData& output);
    static bool rotateArbitrary(const ImageData& input, ImageData& output, 
                                float angle, unsigned char background = 0);
    static bool rotate180InPlace(ImageData& image);
};

// Низкоуровневые CUDA функции
extern "C" {
    void launchRotate90Kernel(const unsigned char* input, unsigned char* output,
                              int width, int height, int channels);
    
    void launchRotate180Kernel(const unsigned char* input, unsigned char* output,
                               int width, int height, int channels);
    
    void launchRotate270Kernel(const unsigned char* input, unsigned char* output,
                               int width, int height, int channels);
    
    void launchRotateArbitraryKernel(const unsigned char* input, unsigned char* output,
                                     int width, int height, int channels,
                                     float angle_rad, unsigned char background);
}

#endif // ROTATION_H

