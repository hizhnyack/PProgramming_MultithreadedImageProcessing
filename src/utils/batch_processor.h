#ifndef BATCH_PROCESSOR_H
#define BATCH_PROCESSOR_H

#include "../core/image_processor.h"
#include <string>
#include <vector>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Пакетная обработка изображений с многопоточностью
class BatchProcessor {
public:
    // Callback для обработки одного изображения
    using ProcessCallback = std::function<bool(ImageData&)>;
    
    // Обработать все изображения в директории
    static bool processDirectory(
        const std::string& input_dir,
        const std::string& output_dir,
        ProcessCallback callback,
        const std::string& extension = ".png"
    );
    
    // Обработать список файлов
    static bool processFiles(
        const std::vector<std::string>& input_files,
        const std::vector<std::string>& output_files,
        ProcessCallback callback
    );
    
    // Обработать один файл
    static bool processFile(
        const std::string& input_file,
        const std::string& output_file,
        ProcessCallback callback
    );
    
    // Получить список файлов в директории
    static std::vector<std::string> getFilesInDirectory(
        const std::string& directory,
        const std::string& extension = ".png"
    );
};

// ============================================================================
// КОНВЕЙЕРНАЯ ОБРАБОТКА (Pipeline) - Ускоренная версия
// ============================================================================

// Структура для элемента конвейера
struct PipelineItem {
    std::string input_file;
    std::string output_file;
    ImageData image;
    size_t index;  // Номер файла для отчетности
};

// Пакетная обработка с конвейером (3 этапа: Загрузка → GPU → Сохранение)
class PipelineBatchProcessor {
public:
    // Callback для обработки одного изображения (с поддержкой CUDA stream)
    using ProcessCallback = std::function<bool(ImageData&, cudaStream_t)>;
    
    // Обработать файлы с конвейером
    static bool processFilesPipelined(
        const std::vector<std::string>& input_files,
        const std::vector<std::string>& output_files,
        ProcessCallback callback,
        int num_streams = 4,      // Количество CUDA streams (параллельная GPU обработка)
        int buffer_size = 10      // Размер буфера между этапами
    );
    
    // Обработать директорию с конвейером
    static bool processDirectoryPipelined(
        const std::string& input_dir,
        const std::string& output_dir,
        ProcessCallback callback,
        const std::string& extension = ".png",
        int num_streams = 4,
        int buffer_size = 10
    );

private:
    // Поток загрузки изображений (читает с диска в память)
    static void loaderThread(
        const std::vector<std::string>& input_files,
        const std::vector<std::string>& output_files,
        std::queue<PipelineItem*>& load_queue,
        std::mutex& queue_mutex,
        std::condition_variable& queue_cv,
        std::atomic<bool>& done_loading,
        int buffer_size
    );
    
    // Поток обработки на GPU (применяет фильтр)
    static void processorThread(
        std::queue<PipelineItem*>& load_queue,
        std::queue<PipelineItem*>& save_queue,
        std::mutex& load_mutex,
        std::mutex& save_mutex,
        std::condition_variable& load_cv,
        std::condition_variable& save_cv,
        ProcessCallback callback,
        cudaStream_t stream,
        std::atomic<bool>& done_loading,
        std::atomic<bool>& done_processing,
        int buffer_size
    );
    
    // Поток сохранения результатов (пишет на диск)
    static void saverThread(
        std::queue<PipelineItem*>& save_queue,
        std::mutex& queue_mutex,
        std::condition_variable& queue_cv,
        std::atomic<bool>& done_processing,
        std::atomic<size_t>& success_count,
        std::atomic<size_t>& fail_count,
        size_t total_files
    );
};

#endif // BATCH_PROCESSOR_H

