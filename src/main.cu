#include <stdio.h>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include "utils/image_loader.h"
#include "utils/batch_processor.h"
#include "filters/grayscale.h"
#include "filters/rotation.h"
#include "filters/blur.h"
#include "filters/cpu_filters.h"

void printUsage(const char* program_name) {
    printf("Usage: %s <command> [options] [--cpu]\n\n", program_name);
    printf("Commands:\n");
    printf("  grayscale <input> <output>          - Convert to grayscale\n");
    printf("  rotate90 <input> <output>           - Rotate 90 degrees\n");
    printf("  rotate180 <input> <output>          - Rotate 180 degrees\n");
    printf("  rotate270 <input> <output>          - Rotate 270 degrees\n");
    printf("  rotateArbitrary <input> <output> <angle> <background> - Rotate by arbitrary angle (0-360)\n");
    printf("  blur <input> <output> <radius>      - Apply blur filter\n");
    printf("  batch <filter> <input_dir> <output_dir>         - Process directory (multithreaded)\n");
    printf("  batch-pipeline <filter> <input_dir> <output_dir> - Process directory (pipeline mode - FASTER!)\n");
    printf("\nOptions:\n");
    printf("  --cpu                                - Use CPU instead of GPU (for comparison)\n");
    printf("\nExample:\n");
    printf("  %s grayscale input.png output.png\n", program_name);
    printf("  %s grayscale input.png output.png --cpu\n", program_name);
    printf("  %s rotateArbitrary input.png output.png 45 128\n", program_name);
    printf("  %s batch grayscale ./images ./output\n", program_name);
}

int main(int argc, char** argv) {
    fprintf(stderr, "[MAIN] Program started with %d arguments\n", argc);
    for (int i = 0; i < argc; i++) {
        fprintf(stderr, "[MAIN]   argv[%d] = %s\n", i, argv[i]);
    }
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Проверяем флаг --cpu
    bool use_cpu = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--cpu") {
            use_cpu = true;
            fprintf(stderr, "[MAIN] CPU mode enabled\n");
            break;
        }
    }
    
    // Инициализируем CUDA только если не используется CPU режим
    ImageProcessor* processor = nullptr;
    if (!use_cpu) {
        fprintf(stderr, "[MAIN] Initializing CUDA...\n");
        try {
            processor = new ImageProcessor();
            fprintf(stderr, "[MAIN] CUDA initialized successfully\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[MAIN] ERROR: CUDA initialization failed: %s\n", e.what());
            return 1;
        }
    } else {
        fprintf(stderr, "[MAIN] Using CPU mode (CUDA not initialized)\n");
    }
    
    std::string command = argv[1];
    fprintf(stderr, "[MAIN] Command: %s\n", command.c_str());
    
    // Single file processing
    if (command == "grayscale" && (argc == 4 || (argc == 5 && use_cpu))) {
        fprintf(stderr, "[MAIN] Processing grayscale: %s -> %s (%s)\n", 
                argv[2], argv[3], use_cpu ? "CPU" : "GPU");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input, !use_cpu)) {
            fprintf(stderr, "[MAIN] Failed to load image\n");
            return 1;
        }
        
        fprintf(stderr, "[PERF] Image loaded: %dx%d, %zu bytes\n", 
                input.width, input.height, input.size_bytes);
        
        bool success = false;
        if (use_cpu) {
            success = CPUFilters::grayscale(input, output);
        } else {
            // Данные уже на GPU (загружены через ImageLoader::load)
            success = GrayscaleFilter::apply(input, output);
            if (success && processor) {
                processor->copyDeviceToHost(output, output);
            }
        }
        
        if (!success) {
            fprintf(stderr, "[MAIN] Failed to apply grayscale filter\n");
            return 1;
        }
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        fprintf(stderr, "[PERF] Total processing time: %ld ms (%.3f s) [%s]\n", 
                duration.count(), duration.count() / 1000.0f, use_cpu ? "CPU" : "GPU");
        
        printf("✓ Grayscale filter applied successfully [%s]\n", use_cpu ? "CPU" : "GPU");
        if (processor) delete processor;
        return 0;
    }
    
    else if (command == "rotate90" && (argc == 4 || (argc == 5 && use_cpu))) {
        fprintf(stderr, "[MAIN] Processing rotate90: %s -> %s (%s)\n", 
                argv[2], argv[3], use_cpu ? "CPU" : "GPU");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input, !use_cpu)) {
            fprintf(stderr, "[MAIN] Failed to load image\n");
            return 1;
        }
        
        fprintf(stderr, "[PERF] Image loaded: %dx%d, %zu bytes\n", 
                input.width, input.height, input.size_bytes);
        
        bool success = false;
        if (use_cpu) {
            success = CPUFilters::rotate90(input, output);
        } else {
            // Данные уже на GPU (загружены через ImageLoader::load)
            success = RotationFilter::rotate90(input, output);
            if (success && processor) {
                processor->copyDeviceToHost(output, output);
            }
        }
        
        if (!success) {
            fprintf(stderr, "[MAIN] Failed to rotate image\n");
            return 1;
        }
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        fprintf(stderr, "[PERF] Total processing time: %ld ms (%.3f s) [%s]\n", 
                duration.count(), duration.count() / 1000.0f, use_cpu ? "CPU" : "GPU");
        
        printf("✓ Image rotated 90° successfully [%s]\n", use_cpu ? "CPU" : "GPU");
        if (processor) delete processor;
        return 0;
    }
    
    else if (command == "rotate180" && (argc == 4 || (argc == 5 && use_cpu))) {
        fprintf(stderr, "[MAIN] Processing rotate180: %s -> %s (%s)\n", 
                argv[2], argv[3], use_cpu ? "CPU" : "GPU");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input, !use_cpu)) {
            return 1;
        }
        
        fprintf(stderr, "[PERF] Image loaded: %dx%d, %zu bytes\n", 
                input.width, input.height, input.size_bytes);
        
        bool success = false;
        if (use_cpu) {
            success = CPUFilters::rotate180(input, output);
        } else {
            // Данные уже на GPU (загружены через ImageLoader::load)
            success = RotationFilter::rotate180(input, output);
            if (success && processor) {
                processor->copyDeviceToHost(output, output);
            }
        }
        
        if (!success) {
            fprintf(stderr, "[MAIN] Failed to rotate image\n");
            return 1;
        }
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        fprintf(stderr, "[PERF] Total processing time: %ld ms (%.3f s) [%s]\n", 
                duration.count(), duration.count() / 1000.0f, use_cpu ? "CPU" : "GPU");
        
        printf("✓ Image rotated 180° successfully [%s]\n", use_cpu ? "CPU" : "GPU");
        if (processor) delete processor;
        return 0;
    }
    
    else if (command == "rotate270" && (argc == 4 || (argc == 5 && use_cpu))) {
        fprintf(stderr, "[MAIN] Processing rotate270: %s -> %s (%s)\n", 
                argv[2], argv[3], use_cpu ? "CPU" : "GPU");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input, !use_cpu)) {
            return 1;
        }
        
        fprintf(stderr, "[PERF] Image loaded: %dx%d, %zu bytes\n", 
                input.width, input.height, input.size_bytes);
        
        bool success = false;
        if (use_cpu) {
            success = CPUFilters::rotate270(input, output);
        } else {
            // Данные уже на GPU (загружены через ImageLoader::load)
            success = RotationFilter::rotate270(input, output);
            if (success && processor) {
                processor->copyDeviceToHost(output, output);
            }
        }
        
        if (!success) {
            fprintf(stderr, "[MAIN] Failed to rotate image\n");
            return 1;
        }
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        fprintf(stderr, "[PERF] Total processing time: %ld ms (%.3f s) [%s]\n", 
                duration.count(), duration.count() / 1000.0f, use_cpu ? "CPU" : "GPU");
        
        printf("✓ Image rotated 270° successfully [%s]\n", use_cpu ? "CPU" : "GPU");
        if (processor) delete processor;
        return 0;
    }
    
    else if (command == "rotateArbitrary" && (argc == 5 || argc == 6 || argc == 7)) {
        // Проверяем, есть ли флаг --cpu (он может быть последним аргументом)
        bool is_cpu = false;
        int angle_idx = 4;
        int bg_idx = 5;
        if (argc == 7 && std::string(argv[6]) == "--cpu") {
            is_cpu = true;
        } else if (argc == 6 && std::string(argv[5]) == "--cpu") {
            is_cpu = true;
            bg_idx = -1;  // background не указан
        }
        
        fprintf(stderr, "[MAIN] Processing rotateArbitrary: %s -> %s (%s)\n", 
                argv[2], argv[3], is_cpu ? "CPU" : "GPU");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ImageData input, output;
        bool cpu_mode = is_cpu || use_cpu;
        if (!ImageLoader::load(argv[2], input, !cpu_mode)) {
            return 1;
        }
        
        fprintf(stderr, "[PERF] Image loaded: %dx%d, %zu bytes\n", 
                input.width, input.height, input.size_bytes);
        
        float angle = atof(argv[angle_idx]);
        unsigned char background = (bg_idx > 0 && argc > bg_idx && !cpu_mode) ? 
                                   (unsigned char)atoi(argv[bg_idx]) : 0;
        
        fprintf(stderr, "[MAIN] Rotation angle: %.1f°, background color: %d\n", angle, background);
        
        bool success = false;
        if (cpu_mode) {
            success = CPUFilters::rotateArbitrary(input, output, angle, background);
        } else {
            // Данные уже на GPU (загружены через ImageLoader::load)
            success = RotationFilter::rotateArbitrary(input, output, angle, background);
            if (success && processor) {
                processor->copyDeviceToHost(output, output);
            }
        }
        
        if (!success) {
            fprintf(stderr, "[MAIN] Failed to rotate image\n");
            return 1;
        }
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        fprintf(stderr, "[PERF] Total processing time: %ld ms (%.3f s) [%s]\n", 
                duration.count(), duration.count() / 1000.0f, (is_cpu || use_cpu) ? "CPU" : "GPU");
        
        printf("✓ Image rotated %.1f° successfully [%s]\n", angle, (is_cpu || use_cpu) ? "CPU" : "GPU");
        if (processor) delete processor;
        return 0;
    }
    
    else if (command == "blur" && (argc == 5 || (argc == 6 && use_cpu))) {
        fprintf(stderr, "[MAIN] Processing blur: %s -> %s (%s)\n", 
                argv[2], argv[3], use_cpu ? "CPU" : "GPU");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input, !use_cpu)) {
            return 1;
        }
        
        fprintf(stderr, "[PERF] Image loaded: %dx%d, %zu bytes\n", 
                input.width, input.height, input.size_bytes);
        
        int radius = atoi(argv[4]);
        if (radius < 1 || radius > 50) {
            fprintf(stderr, "Blur radius must be between 1 and 50\n");
            return 1;
        }
        
        bool success = false;
        if (use_cpu) {
            success = CPUFilters::blurBox(input, output, radius);
        } else {
            // Данные уже на GPU (загружены через ImageLoader::load)
            success = BlurFilter::applyBox(input, output, radius);
            if (success && processor) {
                processor->copyDeviceToHost(output, output);
            }
        }
        
        if (!success) {
            fprintf(stderr, "[MAIN] Failed to apply blur filter\n");
            return 1;
        }
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        fprintf(stderr, "[PERF] Total processing time: %ld ms (%.3f s) [%s]\n", 
                duration.count(), duration.count() / 1000.0f, use_cpu ? "CPU" : "GPU");
        
        printf("✓ Blur filter applied successfully [%s]\n", use_cpu ? "CPU" : "GPU");
        if (processor) delete processor;
        return 0;
    }
    
    // Batch processing
    else if (command == "batch" && argc == 5) {
        std::string filter = argv[2];
        std::string input_dir = argv[3];
        std::string output_dir = argv[4];
        
        BatchProcessor::ProcessCallback callback;
        
        if (filter == "grayscale") {
            callback = [](ImageData& img) {
                return GrayscaleFilter::applyInPlace(img);
            };
        }
        else if (filter == "blur") {
            callback = [](ImageData& img) {
                ImageData output;
                bool result = BlurFilter::applyBox(img, output, 5);
                if (result) {
                    img = output;
                }
                return result;
            };
        }
        else {
            fprintf(stderr, "Unknown filter: %s\n", filter.c_str());
            fprintf(stderr, "Available filters: grayscale, blur\n");
            return 1;
        }
        
        if (!BatchProcessor::processDirectory(input_dir, output_dir, callback)) {
            fprintf(stderr, "Batch processing failed\n");
            return 1;
        }
        
        printf("✓ Batch processing completed\n");
        return 0;
    }
    
    // Pipeline batch processing (FASTER!)
    else if (command == "batch-pipeline" && argc == 5) {
        std::string filter = argv[2];
        std::string input_dir = argv[3];
        std::string output_dir = argv[4];
        
        PipelineBatchProcessor::ProcessCallback callback;
        
        if (filter == "grayscale") {
            callback = [](ImageData& img, cudaStream_t stream) {
                return GrayscaleFilter::apply(img, stream);
            };
        }
        else if (filter == "blur") {
            callback = [](ImageData& img, cudaStream_t stream) {
                return BlurFilter::applyBox(img, 5, stream);
            };
        }
        else if (filter == "rotate90") {
            callback = [](ImageData& img, cudaStream_t stream) {
                return RotationFilter::rotate90(img, stream);
            };
        }
        else if (filter == "rotate180") {
            callback = [](ImageData& img, cudaStream_t stream) {
                return RotationFilter::rotate180(img, stream);
            };
        }
        else if (filter == "rotate270") {
            callback = [](ImageData& img, cudaStream_t stream) {
                return RotationFilter::rotate270(img, stream);
            };
        }
        else {
            fprintf(stderr, "Unknown filter: %s\n", filter.c_str());
            fprintf(stderr, "Available filters: grayscale, blur, rotate90, rotate180, rotate270\n");
            return 1;
        }
        
        // Используем 4 CUDA streams и буфер размером 10
        if (!PipelineBatchProcessor::processDirectoryPipelined(input_dir, output_dir, callback, ".png", 4, 10)) {
            fprintf(stderr, "Pipeline batch processing failed\n");
            return 1;
        }
        
        printf("✓ Pipeline batch processing completed\n");
        return 0;
    }
    
    else {
        fprintf(stderr, "Invalid command or arguments\n\n");
        printUsage(argv[0]);
        return 1;
    }
}

