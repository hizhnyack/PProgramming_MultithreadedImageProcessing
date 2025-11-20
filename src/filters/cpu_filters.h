#ifndef CPU_FILTERS_H
#define CPU_FILTERS_H

#include "../core/image_data.h"
#include <thread>
#include <vector>

// CPU версии фильтров для сравнения производительности с CUDA
namespace CPUFilters {
    // Grayscale
    bool grayscale(const ImageData& input, ImageData& output);
    
    // Rotation
    bool rotate90(const ImageData& input, ImageData& output);
    bool rotate180(const ImageData& input, ImageData& output);
    bool rotate270(const ImageData& input, ImageData& output);
    bool rotateArbitrary(const ImageData& input, ImageData& output, 
                        float angle, unsigned char background = 0);
    
    // Blur
    bool blurBox(const ImageData& input, ImageData& output, int radius);
}

#endif // CPU_FILTERS_H

