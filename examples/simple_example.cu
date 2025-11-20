/**
 * @file simple_example.cu
 * @brief Простой пример использования библиотеки image_processing
 * 
 * Этот пример показывает как:
 * 1. Загрузить изображение
 * 2. Применить фильтр grayscale
 * 3. Сохранить результат
 */

#include "filters/grayscale.h"
#include "utils/image_loader.h"
#include <stdio.h>

int main(int argc, char** argv) {
    printf("=== Простой пример использования библиотеки ===\n\n");
    
    // Проверка аргументов
    if (argc < 3) {
        printf("Использование: %s <input.png> <output.png>\n", argv[0]);
        printf("Пример: %s photo.jpg photo_gray.jpg\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    
    printf("📸 Загрузка изображения: %s\n", input_file);
    
    // Шаг 1: Загрузка изображения
    ImageData image;
    if (!ImageLoader::load(input_file, image)) {
        printf("❌ Ошибка загрузки изображения!\n");
        return 1;
    }
    
    printf("✓ Изображение загружено: %dx%d, %d каналов\n", 
           image.width, image.height, image.channels);
    
    // Шаг 2: Применение фильтра grayscale
    printf("\n🎨 Применение фильтра grayscale...\n");
    
    if (!GrayscaleFilter::applyInPlace(image)) {
        printf("❌ Ошибка применения фильтра!\n");
        return 1;
    }
    
    printf("✓ Фильтр применен\n");
    
    // Шаг 3: Копирование результата с GPU на CPU
    printf("\n💾 Копирование результата с GPU...\n");
    
    cudaError_t err = cudaMemcpy(image.data, image.gpu_data, 
                                  image.size_bytes, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        printf("❌ Ошибка копирования: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("✓ Данные скопированы\n");
    
    // Шаг 4: Сохранение результата
    printf("\n💾 Сохранение результата: %s\n", output_file);
    
    if (!ImageLoader::save(output_file, image)) {
        printf("❌ Ошибка сохранения изображения!\n");
        return 1;
    }
    
    printf("✓ Результат сохранен\n");
    
    printf("\n🎉 Готово! Проверь файл: %s\n", output_file);
    
    return 0;
}

