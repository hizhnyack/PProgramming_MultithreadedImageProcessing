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

// ============================================================================
// КОНВЕЙЕРНАЯ ОБРАБОТКА (Pipeline Implementation)
// ============================================================================

// Поток 1: ЗАГРУЗКА изображений с диска
void PipelineBatchProcessor::loaderThread(
    const std::vector<std::string>& input_files,
    const std::vector<std::string>& output_files,
    std::queue<PipelineItem*>& load_queue,
    std::mutex& queue_mutex,
    std::condition_variable& queue_cv,
    std::atomic<bool>& done_loading,
    int buffer_size
) {
    for (size_t i = 0; i < input_files.size(); ++i) {
        // Создаем элемент конвейера
        PipelineItem* item = new PipelineItem();
        item->input_file = input_files[i];
        item->output_file = output_files[i];
        item->index = i;
        
        // Загружаем изображение
        if (!ImageLoader::load(item->input_file, item->image)) {
            std::lock_guard<std::mutex> lock(console_mutex);
            fprintf(stderr, "[Loader] Failed to load: %s\n", item->input_file.c_str());
            delete item;
            continue;
        }
        
        // Ждем пока в очереди появится место
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&]() { 
                return load_queue.size() < (size_t)buffer_size; 
            });
            
            // Кладем в очередь
            load_queue.push(item);
            
            {
                std::lock_guard<std::mutex> console_lock(console_mutex);
                printf("[Loader] Loaded [%zu/%zu]: %s (queue: %zu)\n", 
                       i + 1, input_files.size(), item->input_file.c_str(), load_queue.size());
            }
        }
        
        // Сигнализируем что есть работа
        queue_cv.notify_all();
    }
    
    done_loading = true;
    queue_cv.notify_all();  // Будим всех кто ждет
    
    std::lock_guard<std::mutex> lock(console_mutex);
    printf("[Loader] Finished loading all files\n");
}

// Поток 2: ОБРАБОТКА на GPU
void PipelineBatchProcessor::processorThread(
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
) {
    while (true) {
        PipelineItem* item = nullptr;
        
        // Берем элемент из очереди загрузки
        {
            std::unique_lock<std::mutex> lock(load_mutex);
            load_cv.wait(lock, [&]() { 
                return !load_queue.empty() || done_loading; 
            });
            
            if (load_queue.empty() && done_loading) {
                break;  // Работа закончена
            }
            
            if (!load_queue.empty()) {
                item = load_queue.front();
                load_queue.pop();
            }
        }
        
        // Сигнализируем что место освободилось
        load_cv.notify_all();
        
        if (!item) continue;
        
        // Обрабатываем на GPU (с использованием stream)
        bool success = callback(item->image, stream);
        
        if (!success) {
            std::lock_guard<std::mutex> lock(console_mutex);
            fprintf(stderr, "[GPU] Failed to process: %s\n", item->input_file.c_str());
            item->image.cleanup();
            delete item;
            continue;
        }
        
        // Копируем результат обратно на host (асинхронно через stream)
        if (item->image.gpu_data && item->image.data) {
            cudaMemcpyAsync(item->image.data, item->image.gpu_data, 
                           item->image.size_bytes, cudaMemcpyDeviceToHost, stream);
        }
        
        // Синхронизируем stream перед сохранением
        cudaStreamSynchronize(stream);
        
        {
            std::lock_guard<std::mutex> lock(console_mutex);
            printf("[GPU] Processed [%zu]: %s\n", item->index + 1, item->input_file.c_str());
        }
        
        // Ждем пока в очереди сохранения появится место
        {
            std::unique_lock<std::mutex> lock(save_mutex);
            save_cv.wait(lock, [&]() { 
                return save_queue.size() < (size_t)buffer_size; 
            });
            
            save_queue.push(item);
        }
        
        // Сигнализируем что есть что сохранять
        save_cv.notify_all();
    }
    
    std::lock_guard<std::mutex> lock(console_mutex);
    printf("[GPU] Finished processing\n");
}

// Поток 3: СОХРАНЕНИЕ результатов на диск
void PipelineBatchProcessor::saverThread(
    std::queue<PipelineItem*>& save_queue,
    std::mutex& queue_mutex,
    std::condition_variable& queue_cv,
    std::atomic<bool>& done_processing,
    std::atomic<size_t>& success_count,
    std::atomic<size_t>& fail_count,
    size_t total_files
) {
    while (true) {
        PipelineItem* item = nullptr;
        
        // Берем элемент из очереди сохранения
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&]() { 
                return !save_queue.empty() || done_processing; 
            });
            
            if (save_queue.empty() && done_processing) {
                break;  // Работа закончена
            }
            
            if (!save_queue.empty()) {
                item = save_queue.front();
                save_queue.pop();
            }
        }
        
        // Сигнализируем что место освободилось
        queue_cv.notify_all();
        
        if (!item) continue;
        
        // Сохраняем результат
        bool success = ImageLoader::save(item->output_file, item->image);
        
        if (success) {
            success_count++;
            std::lock_guard<std::mutex> lock(console_mutex);
            printf("[Saver] Saved [%zu/%zu]: %s\n", 
                   item->index + 1, total_files, item->output_file.c_str());
        } else {
            fail_count++;
            std::lock_guard<std::mutex> lock(console_mutex);
            fprintf(stderr, "[Saver] Failed to save: %s\n", item->output_file.c_str());
        }
        
        // Очищаем память
        item->image.cleanup();
        delete item;
    }
    
    std::lock_guard<std::mutex> lock(console_mutex);
    printf("[Saver] Finished saving\n");
}

// Главная функция конвейерной обработки
bool PipelineBatchProcessor::processFilesPipelined(
    const std::vector<std::string>& input_files,
    const std::vector<std::string>& output_files,
    ProcessCallback callback,
    int num_streams,
    int buffer_size
) {
    if (input_files.size() != output_files.size()) {
        fprintf(stderr, "Input and output file lists must have the same size\n");
        return false;
    }
    
    if (input_files.empty()) {
        printf("No files to process\n");
        return true;
    }
    
    printf("\n=== Pipeline Batch Processing ===\n");
    printf("Files: %zu\n", input_files.size());
    printf("GPU Streams: %d\n", num_streams);
    printf("Buffer Size: %d\n", buffer_size);
    printf("==================================\n\n");
    
    // Создаем CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Очереди и синхронизация
    std::queue<PipelineItem*> load_queue;   // Загруженные изображения
    std::queue<PipelineItem*> save_queue;   // Обработанные изображения
    
    std::mutex load_mutex, save_mutex;
    std::condition_variable load_cv, save_cv;
    
    std::atomic<bool> done_loading(false);
    std::atomic<bool> done_processing(false);
    std::atomic<size_t> success_count(0);
    std::atomic<size_t> fail_count(0);
    
    // Запускаем поток загрузки
    std::thread loader(loaderThread,
        std::cref(input_files),
        std::cref(output_files),
        std::ref(load_queue),
        std::ref(load_mutex),
        std::ref(load_cv),
        std::ref(done_loading),
        buffer_size
    );
    
    // Запускаем потоки обработки (по одному на каждый stream)
    std::vector<std::thread> processors;
    for (int i = 0; i < num_streams; ++i) {
        processors.emplace_back(processorThread,
            std::ref(load_queue),
            std::ref(save_queue),
            std::ref(load_mutex),
            std::ref(save_mutex),
            std::ref(load_cv),
            std::ref(save_cv),
            callback,
            streams[i],
            std::ref(done_loading),
            std::ref(done_processing),
            buffer_size
        );
    }
    
    // Запускаем поток сохранения
    std::thread saver(saverThread,
        std::ref(save_queue),
        std::ref(save_mutex),
        std::ref(save_cv),
        std::ref(done_processing),
        std::ref(success_count),
        std::ref(fail_count),
        input_files.size()
    );
    
    // Ждем завершения загрузки
    loader.join();
    
    // Ждем завершения обработки
    for (auto& thread : processors) {
        thread.join();
    }
    done_processing = true;
    save_cv.notify_all();
    
    // Ждем завершения сохранения
    saver.join();
    
    // Очищаем CUDA streams
    for (auto stream : streams) {
        cudaStreamDestroy(stream);
    }
    
    printf("\n=== Pipeline Complete ===\n");
    printf("Success: %zu\n", success_count.load());
    printf("Failed: %zu\n", fail_count.load());
    printf("Total: %zu\n", input_files.size());
    
    return fail_count == 0;
}

// Обработка директории с конвейером
bool PipelineBatchProcessor::processDirectoryPipelined(
    const std::string& input_dir,
    const std::string& output_dir,
    ProcessCallback callback,
    const std::string& extension,
    int num_streams,
    int buffer_size
) {
    // Получаем список файлов
    std::vector<std::string> input_files = BatchProcessor::getFilesInDirectory(input_dir, extension);
    
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
        size_t pos = input_file.rfind('/');
        std::string filename = (pos != std::string::npos) 
            ? input_file.substr(pos + 1) 
            : input_file;
        
        std::string output_file = output_dir;
        if (output_file.back() != '/') output_file += '/';
        output_file += filename;
        
        output_files.push_back(output_file);
    }
    
    return processFilesPipelined(input_files, output_files, callback, num_streams, buffer_size);
}

