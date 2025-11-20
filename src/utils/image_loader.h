#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "../core/image_processor.h"
#include <string>

// Загрузка и сохранение изображений (PNG, JPG, BMP)
class ImageLoader {
public:
    // Загрузить изображение из файла
    // use_gpu: если false, данные не копируются на GPU (для CPU режима)
    static bool load(const std::string& filename, ImageData& image, bool use_gpu = true);
    
    // Сохранить изображение в файл
    static bool save(const std::string& filename, const ImageData& image);
    
    // Получить информацию об изображении без загрузки
    static bool getInfo(const std::string& filename, int& width, int& height, int& channels);
};

#endif // IMAGE_LOADER_H

