#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

// Макрос для проверки CUDA ошибок с выходом из программы (для критических ошибок)
#define CUDA_CHECK_CRITICAL(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Макрос для проверки CUDA ошибок с возвратом false (для функций фильтров)
#define CUDA_CHECK_RETURN(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return false; \
        } \
    } while(0)

// Для обратной совместимости - по умолчанию критическая проверка
#ifndef CUDA_CHECK
#define CUDA_CHECK CUDA_CHECK_CRITICAL
#endif

#endif // CUDA_UTILS_H

