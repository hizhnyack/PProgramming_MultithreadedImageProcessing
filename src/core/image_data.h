#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include <cstddef>

// Структура данных изображения без зависимостей от CUDA
struct ImageData {
    unsigned char* data; //host memory pointer
    int width;
    int height;
    int channels;
    size_t size_bytes;
    
    // GPU memory pointer (только для CUDA версий)
    // В CPU версиях всегда nullptr
    unsigned char* gpu_data;
    
    ImageData() : data(nullptr), width(0), height(0), channels(0), 
                  size_bytes(0), gpu_data(nullptr) {}
    
    ~ImageData() {
        cleanup();
    }
    
    void cleanup() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
        // gpu_data очищается в CUDA версиях через cleanupGPU()
        // Для CPU версий просто обнуляем указатель
        // Для CUDA версий cleanupGPU() вызывается автоматически в деструкторе
        // через условную компиляцию или явный вызов
        if (gpu_data) {
            // В CPU версиях gpu_data должен быть nullptr
            // В CUDA версиях cleanupGPU() должен быть вызван отдельно
            gpu_data = nullptr;
        }
    }
    
    // Метод для очистки GPU памяти (вызывается из CUDA кода)
    // Реализация в image_processor.cu
    void cleanupGPU();

    int getPixelCount() const { return width * height * channels; }
};

#endif // IMAGE_DATA_H

