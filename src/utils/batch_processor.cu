#include "batch_processor.h"
#include "image_loader.h"
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <thread>
#include <mutex>
#include <atomic>

// Mutex для безопасного вывода в консоль
static std::mutex console_mutex;

bool BatchProcessor::processFile(
    const std::string& input_file,
    const std::string& output_file,
    ProcessCallback callback
) {
    ImageData image;
    
    // Загружаем изображение
    if (!ImageLoader::load(input_file, image)) {
        return false;
    }
    
    // Применяем обработку
    bool success = callback(image);
    
    if (!success) {
        std::lock_guard<std::mutex> lock(console_mutex);
        fprintf(stderr, "Failed to process: %s\n", input_file.c_str());
        image.cleanup();
        return false;
    }
    
    // Копируем результат обратно на host если нужно
    if (image.gpu_data && image.data) {
        cudaMemcpy(image.data, image.gpu_data, image.size_bytes, cudaMemcpyDeviceToHost);
    }
    
    // Сохраняем результат
    success = ImageLoader::save(output_file, image);
    
    image.cleanup();
    return success;
}

bool BatchProcessor::processFiles(
    const std::vector<std::string>& input_files,
    const std::vector<std::string>& output_files,
    ProcessCallback callback
) {
    if (input_files.size() != output_files.size()) {
        fprintf(stderr, "Input and output file lists must have the same size\n");
        return false;
    }
    
    if (input_files.empty()) {
        printf("No files to process\n");
        return true;
    }
    
    // Количество потоков = количество CPU ядер
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    printf("Processing %zu files using %u threads...\n", input_files.size(), num_threads);
    
    std::atomic<size_t> current_index(0);
    std::atomic<size_t> success_count(0);
    std::atomic<size_t> fail_count(0);
    
    auto worker = [&]() {
        while (true) {
            size_t index = current_index.fetch_add(1);
            if (index >= input_files.size()) break;
            
            {
                std::lock_guard<std::mutex> lock(console_mutex);
                printf("[%zu/%zu] Processing: %s\n", 
                       index + 1, input_files.size(), input_files[index].c_str());
            }
            
            if (processFile(input_files[index], output_files[index], callback)) {
                success_count++;
            } else {
                fail_count++;
            }
        }
    };
    
    // Запускаем потоки
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    
    // Ждем завершения
    for (auto& thread : threads) {
        thread.join();
    }
    
    printf("\n=== Batch Processing Complete ===\n");
    printf("Success: %zu\n", success_count.load());
    printf("Failed: %zu\n", fail_count.load());
    printf("Total: %zu\n", input_files.size());
    
    return fail_count == 0;
}

std::vector<std::string> BatchProcessor::getFilesInDirectory(
    const std::string& directory,
    const std::string& extension
) {
    std::vector<std::string> files;
    
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", directory.c_str());
        return files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Пропускаем . и ..
        if (filename == "." || filename == "..") continue;
        
        // Проверяем расширение
        if (!extension.empty()) {
            size_t pos = filename.rfind(extension);
            if (pos == std::string::npos || pos != filename.length() - extension.length()) {
                continue;
            }
        }
        
        // Полный путь
        std::string full_path = directory;
        if (full_path.back() != '/') full_path += '/';
        full_path += filename;
        
        // Проверяем что это файл
        struct stat st;
        if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            files.push_back(full_path);
        }
    }
    
    closedir(dir);
    return files;
}

bool BatchProcessor::processDirectory(
    const std::string& input_dir,
    const std::string& output_dir,
    ProcessCallback callback,
    const std::string& extension
) {
    // Получаем список файлов
    std::vector<std::string> input_files = getFilesInDirectory(input_dir, extension);
    
    if (input_files.empty()) {
        printf("No files found in directory: %s\n", input_dir.c_str());
        return true;
    }
    
    // Создаем выходную директорию если не существует
    struct stat st;
    if (stat(output_dir.c_str(), &st) != 0) {
        mkdir(output_dir.c_str(), 0755);
    }
    
    // Формируем список выходных файлов
    std::vector<std::string> output_files;
    for (const auto& input_file : input_files) {
        // Извлекаем имя файла
        size_t pos = input_file.rfind('/');
        std::string filename = (pos != std::string::npos) 
            ? input_file.substr(pos + 1) 
            : input_file;
        
        // Формируем выходной путь
        std::string output_file = output_dir;
        if (output_file.back() != '/') output_file += '/';
        output_file += filename;
        
        output_files.push_back(output_file);
    }
    
    return processFiles(input_files, output_files, callback);
}

