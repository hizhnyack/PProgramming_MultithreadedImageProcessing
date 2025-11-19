#ifndef BATCH_PROCESSOR_H
#define BATCH_PROCESSOR_H

#include "../core/image_processor.h"
#include <string>
#include <vector>
#include <functional>

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

#endif // BATCH_PROCESSOR_H

