#include "../src/core/parallel_processor.h"
#include "../src/core/image_processor.h"
#include <iostream>
#include <vector>
#include <chrono>

/**
 * @brief Тест параллельной обработки изображений
 * 
 * Создает несколько тестовых изображений и обрабатывает их:
 * 1. Последовательно (по одному)
 * 2. Параллельно (используя CUDA streams)
 * 
 * Сравнивает производительность и корректность результатов.
 */

// Создание тестового изображения с градиентом
std::vector<unsigned char> createTestImage(int width, int height, int seed) {
    std::vector<unsigned char> image(width * height * 3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            // Создаем градиент с учетом seed для разнообразия
            image[idx + 0] = (x * 255 / width + seed) % 256;        // R
            image[idx + 1] = (y * 255 / height + seed) % 256;       // G
            image[idx + 2] = ((x + y) * 255 / (width + height) + seed) % 256; // B
        }
    }
    
    return image;
}

// Последовательная обработка для сравнения
double processSequentially(
    const std::vector<std::vector<unsigned char>>& inputImages,
    std::vector<std::vector<unsigned char>>& outputImages,
    int width,
    int height
) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    outputImages.resize(inputImages.size());
    
    for (size_t i = 0; i < inputImages.size(); i++) {
        outputImages[i].resize(width * height * 3);
        
        // Простое преобразование в grayscale на CPU
        for (int j = 0; j < width * height; j++) {
            unsigned char r = inputImages[i][j * 3 + 0];
            unsigned char g = inputImages[i][j * 3 + 1];
            unsigned char b = inputImages[i][j * 3 + 2];
            
            unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
            
            outputImages[i][j * 3 + 0] = gray;
            outputImages[i][j * 3 + 1] = gray;
            outputImages[i][j * 3 + 2] = gray;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    return duration.count();
}

// Проверка корректности результата (первые несколько пикселей)
bool verifyResults(
    const std::vector<unsigned char>& result,
    int width,
    int height
) {
    // Проверяем, что это действительно grayscale (R == G == B)
    for (int i = 0; i < std::min(100, width * height); i++) {
        unsigned char r = result[i * 3 + 0];
        unsigned char g = result[i * 3 + 1];
        unsigned char b = result[i * 3 + 2];
        
        if (r != g || g != b) {
            std::cerr << "Error: Pixel " << i << " is not grayscale: "
                      << "R=" << (int)r << " G=" << (int)g << " B=" << (int)b << std::endl;
            return false;
        }
    }
    
    return true;
}

int main(int argc, char** argv) {
    std::cout << "=== Тест параллельной обработки изображений ===" << std::endl;
    std::cout << std::endl;
    
    // Параметры теста
    int width = 1920;
    int height = 1080;
    int numImages = 8;
    
    if (argc >= 2) {
        numImages = std::atoi(argv[1]);
    }
    
    std::cout << "Параметры теста:" << std::endl;
    std::cout << "  Размер изображения: " << width << "x" << height << std::endl;
    std::cout << "  Количество изображений: " << numImages << std::endl;
    std::cout << std::endl;
    
    // Шаг 1: Создание тестовых изображений
    std::cout << "📸 Создание тестовых изображений..." << std::endl;
    std::vector<std::vector<unsigned char>> testImages;
    
    for (int i = 0; i < numImages; i++) {
        testImages.push_back(createTestImage(width, height, i * 42));
    }
    
    std::cout << "✓ Создано " << testImages.size() << " изображений" << std::endl;
    std::cout << std::endl;
    
    // Шаг 2: Инициализация ParallelProcessor
    std::cout << "🔧 Инициализация ParallelProcessor..." << std::endl;
    
    ParallelProcessor::ParallelConfig config;
    config.maxConcurrentStreams = 4;
    config.blockSize = 16;
    config.enableTiming = true;
    config.verbose = true;
    
    if (!ParallelProcessor::initialize(config)) {
        std::cerr << "❌ Ошибка инициализации ParallelProcessor" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    ParallelProcessor::printGPUInfo();
    std::cout << std::endl;
    
    // Шаг 3: Параллельная обработка
    std::cout << "⚡ Запуск параллельной обработки..." << std::endl;
    
    std::vector<std::vector<unsigned char>> parallelResults;
    auto parallelStats = ParallelProcessor::processBatchParallel(
        testImages,
        parallelResults,
        width,
        height,
        "grayscale"
    );
    
    std::cout << std::endl;
    std::cout << "📊 Результаты параллельной обработки:" << std::endl;
    parallelStats.print();
    std::cout << std::endl;
    
    // Шаг 4: Последовательная обработка для сравнения
    std::cout << "🐌 Запуск последовательной обработки (для сравнения)..." << std::endl;
    
    std::vector<std::vector<unsigned char>> sequentialResults;
    double sequentialTime = processSequentially(
        testImages,
        sequentialResults,
        width,
        height
    );
    
    std::cout << "✓ Последовательная обработка завершена за " << sequentialTime << " ms" << std::endl;
    std::cout << std::endl;
    
    // Шаг 5: Сравнение производительности
    std::cout << "🏁 Сравнение производительности:" << std::endl;
    std::cout << "  Последовательно: " << sequentialTime << " ms" << std::endl;
    std::cout << "  Параллельно:     " << parallelStats.totalTimeMs << " ms" << std::endl;
    
    if (parallelStats.success && sequentialTime > 0) {
        double speedup = sequentialTime / parallelStats.totalTimeMs;
        std::cout << "  Ускорение:       " << speedup << "x" << std::endl;
        
        if (speedup > 1.0) {
            std::cout << "  ✅ Параллельная обработка быстрее!" << std::endl;
        } else {
            std::cout << "  ⚠️  Последовательная обработка быстрее (возможно, overhead)" << std::endl;
        }
    }
    std::cout << std::endl;
    
    // Шаг 6: Проверка корректности
    std::cout << "🔍 Проверка корректности результатов..." << std::endl;
    
    bool allCorrect = true;
    for (size_t i = 0; i < parallelResults.size(); i++) {
        if (!verifyResults(parallelResults[i], width, height)) {
            std::cerr << "❌ Ошибка в изображении " << i << std::endl;
            allCorrect = false;
        }
    }
    
    if (allCorrect) {
        std::cout << "✅ Все результаты корректны!" << std::endl;
    }
    std::cout << std::endl;
    
    // Очистка
    ParallelProcessor::cleanup();
    
    std::cout << "=== Тест завершен ===" << std::endl;
    
    return (parallelStats.success && allCorrect) ? 0 : 1;
}

