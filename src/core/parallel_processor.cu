#include "parallel_processor.h"
#include <algorithm>
#include <cstring>

// Инициализация статических переменных
ParallelProcessor::ParallelConfig ParallelProcessor::s_config;
bool ParallelProcessor::s_initialized = false;

/**
 * @brief CUDA kernel для преобразования в grayscale
 */
__global__ void ParallelProcessor::grayscaleKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char r = input[idx * 3 + 0];
        unsigned char g = input[idx * 3 + 1];
        unsigned char b = input[idx * 3 + 2];
        
        // Формула преобразования в grayscale
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        
        output[idx * 3 + 0] = gray;
        output[idx * 3 + 1] = gray;
        output[idx * 3 + 2] = gray;
    }
}

/**
 * @brief CUDA kernel для инвертирования цветов
 */
__global__ void ParallelProcessor::invertKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx * 3 + 0] = 255 - input[idx * 3 + 0];
        output[idx * 3 + 1] = 255 - input[idx * 3 + 1];
        output[idx * 3 + 2] = 255 - input[idx * 3 + 2];
    }
}

bool ParallelProcessor::initialize(const ParallelConfig& config) {
    s_config = config;
    
    // Проверяем доступность CUDA
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No CUDA devices found" << std::endl;
        return false;
    }
    
    // Получаем информацию о GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    logMessage("ParallelProcessor initialized successfully");
    logMessage("GPU: " + std::string(prop.name));
    logMessage("Max concurrent streams: " + std::to_string(s_config.maxConcurrentStreams));
    logMessage("Block size: " + std::to_string(s_config.blockSize) + "x" + std::to_string(s_config.blockSize));
    
    s_initialized = true;
    return true;
}

ParallelProcessor::ProcessingStats ParallelProcessor::processBatchParallel(
    const std::vector<std::vector<unsigned char>>& inputImages,
    std::vector<std::vector<unsigned char>>& outputImages,
    int width,
    int height,
    const std::string& filterType
) {
    ProcessingStats stats;
    stats.totalImages = inputImages.size();
    
    if (!s_initialized) {
        std::cerr << "Error: ParallelProcessor not initialized. Call initialize() first." << std::endl;
        return stats;
    }
    
    if (stats.totalImages == 0) {
        std::cerr << "Error: No input images provided" << std::endl;
        return stats;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Создаем CUDA streams для параллельной обработки
        int numStreams = std::min(s_config.maxConcurrentStreams, static_cast<int>(stats.totalImages));
        std::vector<cudaStream_t> streams;
        
        if (!setupCUDAStreams(streams, numStreams)) {
            return stats;
        }
        
        size_t imageSize = width * height * 3 * sizeof(unsigned char);
        
        // Выделяем память на GPU для всех изображений
        std::vector<unsigned char*> d_inputs(stats.totalImages, nullptr);
        std::vector<unsigned char*> d_outputs(stats.totalImages, nullptr);
        
        // Копируем данные на GPU асинхронно
        for (size_t i = 0; i < stats.totalImages; i++) {
            cudaMalloc(&d_inputs[i], imageSize);
            cudaMalloc(&d_outputs[i], imageSize);
            
            int streamIndex = i % numStreams;
            cudaMemcpyAsync(d_inputs[i], inputImages[i].data(), imageSize,
                           cudaMemcpyHostToDevice, streams[streamIndex]);
        }
        
        // Запускаем обработку в параллельных потоках
        for (size_t i = 0; i < stats.totalImages; i++) {
            int streamIndex = i % numStreams;
            applyFilterToImage(d_inputs[i], d_outputs[i], width, height, filterType, streams[streamIndex]);
        }
        
        // Копируем результаты обратно на CPU
        outputImages.resize(stats.totalImages);
        for (size_t i = 0; i < stats.totalImages; i++) {
            outputImages[i].resize(width * height * 3);
            int streamIndex = i % numStreams;
            cudaMemcpyAsync(outputImages[i].data(), d_outputs[i], imageSize,
                           cudaMemcpyDeviceToHost, streams[streamIndex]);
        }
        
        // Синхронизируем все потоки
        for (auto& stream : streams) {
            cudaStreamSynchronize(stream);
        }
        
        // Проверяем ошибки CUDA
        cudaError_t finalError = cudaGetLastError();
        if (finalError != cudaSuccess) {
            std::cerr << "CUDA error during processing: " << cudaGetErrorString(finalError) << std::endl;
            stats.success = false;
        } else {
            stats.processedImages = stats.totalImages;
            stats.success = true;
        }
        
        // Освобождаем GPU память
        for (size_t i = 0; i < stats.totalImages; i++) {
            if (d_inputs[i]) cudaFree(d_inputs[i]);
            if (d_outputs[i]) cudaFree(d_outputs[i]);
        }
        
        cleanupCUDAStreams(streams);
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during parallel processing: " << e.what() << std::endl;
        stats.success = false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    stats.totalTimeMs = duration.count();
    if (stats.processedImages > 0) {
        stats.averageTimePerImageMs = stats.totalTimeMs / static_cast<double>(stats.processedImages);
    }
    
    if (s_config.verbose && stats.success) {
        logMessage("Successfully processed " + std::to_string(stats.processedImages) +
                  " images in " + std::to_string(stats.totalTimeMs) + "ms");
    }
    
    return stats;
}

bool ParallelProcessor::setupCUDAStreams(std::vector<cudaStream_t>& streams, int count) {
    streams.resize(count);
    for (int i = 0; i < count; i++) {
        cudaError_t error = cudaStreamCreate(&streams[i]);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream " << i << ": " << cudaGetErrorString(error) << std::endl;
            cleanupCUDAStreams(streams);
            return false;
        }
    }
    return true;
}

void ParallelProcessor::cleanupCUDAStreams(std::vector<cudaStream_t>& streams) {
    for (auto& stream : streams) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    streams.clear();
}

void ParallelProcessor::applyFilterToImage(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    const std::string& filterType,
    cudaStream_t stream
) {
    dim3 blockDim(s_config.blockSize, s_config.blockSize);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    if (filterType == "grayscale") {
        grayscaleKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, width, height);
    }
    else if (filterType == "invert") {
        invertKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, width, height);
    }
    else {
        std::cerr << "Warning: Unknown filter type '" << filterType << "', using grayscale" << std::endl;
        grayscaleKernel<<<gridDim, blockDim, 0, stream>>>(d_input, d_output, width, height);
    }
}

void ParallelProcessor::printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "CUDA Devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  Async Engines: " << prop.asyncEngineCount << std::endl;
    }
}

void ParallelProcessor::cleanup() {
    s_initialized = false;
    logMessage("ParallelProcessor cleanup completed");
}

void ParallelProcessor::logMessage(const std::string& message) {
    if (s_config.verbose) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "[ParallelProcessor] " << message << std::endl;
    }
}

bool ParallelProcessor::checkCUDAError(const std::string& operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error during " << operation << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}
