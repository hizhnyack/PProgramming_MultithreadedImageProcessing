#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

// Forward declarations для CUDA kernels
__global__ void parallelGrayscaleKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
);

__global__ void parallelInvertKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
);

/**
 * @brief Класс для параллельной обработки изображений с использованием CUDA streams
 *
 * Этот класс позволяет обрабатывать несколько изображений одновременно,
 * используя асинхронные операции и множественные CUDA потоки.
 */
class ParallelProcessor {
public:
    /**
     * @brief Статистика выполнения обработки
     */
    struct ProcessingStats {
        size_t totalImages;           ///< Общее количество изображений
        size_t processedImages;       ///< Успешно обработанные изображения
        double totalTimeMs;           ///< Общее время выполнения (мс)
        double averageTimePerImageMs; ///< Среднее время на изображение (мс)
        bool success;                 ///< Флаг успешного выполнения

        ProcessingStats() : totalImages(0), processedImages(0),
                           totalTimeMs(0.0), averageTimePerImageMs(0.0),
                           success(false) {}
        
        /// Вывод статистики в читаемом формате
        void print() const {
            std::cout << "=== Processing Statistics ===" << std::endl;
            std::cout << "Total images: " << totalImages << std::endl;
            std::cout << "Processed: " << processedImages << std::endl;
            std::cout << "Total time: " << totalTimeMs << " ms" << std::endl;
            std::cout << "Average per image: " << averageTimePerImageMs << " ms" << std::endl;
            std::cout << "Success: " << (success ? "Yes" : "No") << std::endl;
        }
    };

    /**
     * @brief Конфигурация параллельной обработки
     */
    struct ParallelConfig {
        int maxConcurrentStreams = 4;    ///< Максимум одновременных потоков CUDA
        int blockSize = 16;              ///< Размер блока для kernel (16x16)
        bool enableTiming = true;        ///< Включить замер времени
        bool verbose = false;            ///< Подробный вывод

        ParallelConfig() = default;
        
        /// Конструктор с параметрами
        ParallelConfig(int streams, int block, bool timing, bool verb)
            : maxConcurrentStreams(streams), blockSize(block),
              enableTiming(timing), verbose(verb) {}
    };

    /**
     * @brief Инициализация параллельного процессора
     * @param config Конфигурация обработки
     * @return true если инициализация успешна, false в противном случае
     */
    static bool initialize(const ParallelConfig& config);

    /**
     * @brief Параллельная обработка batch изображений одним фильтром
     * @param inputImages Входные изображения
     * @param outputImages Выходные изображения (будет изменен)
     * @param width Ширина изображений
     * @param height Высота изображений
     * @param filterType Тип фильтра ("grayscale", "invert")
     * @return Статистика выполнения
     */
    static ProcessingStats processBatchParallel(
        const std::vector<std::vector<unsigned char>>& inputImages,
        std::vector<std::vector<unsigned char>>& outputImages,
        int width,
        int height,
        const std::string& filterType = "grayscale"
    );

    /**
     * @brief Получить информацию о GPU устройствах
     */
    static void printGPUInfo();

    /**
     * @brief Очистка ресурсов
     */
    static void cleanup();

    /**
     * @brief Проверка инициализации
     * @return true если процессор инициализирован
     */
    static bool isInitialized() { return s_initialized; }

    /**
     * @brief Получить текущую конфигурацию
     * @return Текущая конфигурация
     */
    static ParallelConfig getConfig() { return s_config; }

private:
    static ParallelConfig s_config;      ///< Текущая конфигурация
    static bool s_initialized;           ///< Флаг инициализации

    // Вспомогательные методы
    static bool setupCUDAStreams(std::vector<cudaStream_t>& streams, int count);
    static void cleanupCUDAStreams(std::vector<cudaStream_t>& streams);

    static void applyFilterToImage(
        unsigned char* d_input,
        unsigned char* d_output,
        int width,
        int height,
        const std::string& filterType,
        cudaStream_t stream
    );

    static void logMessage(const std::string& message);
    static bool checkCUDAError(const std::string& operation);
};

#endif // PARALLEL_PROCESSOR_H
