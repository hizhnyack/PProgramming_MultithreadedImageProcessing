cmake_minimum_required(VERSION 3.18)

# Для WSL - устанавливаем архитектуру ДО project()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Используем архитектуру 86 (Ampere) для совместимости, работает на RTX 4060
set(CMAKE_CUDA_ARCHITECTURES 86)

project(MultithreadedImageProcessing CUDA CXX)

# Обход проверки версии компилятора (для GCC 13 в WSL и VS 2026 в Windows)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
add_compile_definitions(__NV_NO_HOST_COMPILER_CHECK=1)

find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

set(CORE_SOURCES
    src/core/image_processor.cu
    src/core/cuda_kernels.cu
    src/core/parallel_processor.cu 
)

set(FILTER_SOURCES
    src/filters/grayscale.cu
    src/filters/rotation.cu
    src/filters/blur.cu
)

set(UTILS_SOURCES
    src/utils/image_loader.cu
    src/utils/batch_processor.cu
)

add_executable(image_processor
    src/main.cu
    ${CORE_SOURCES}
    ${FILTER_SOURCES}
    ${UTILS_SOURCES}
)

set_target_properties(image_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(image_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler -Wall
    >
)

# Линковка с C++ стандартной библиотекой и математической библиотекой
target_link_libraries(image_processor PRIVATE
    stdc++
    m
    cudart
)

# Отдельные тесты для каждого фильтра
add_executable(test_grayscale
    tests/test_grayscale.cu
    ${CORE_SOURCES}
    src/filters/grayscale.cu
)
target_link_libraries(test_grayscale PRIVATE stdc++ m cudart)

add_executable(test_rotation
    tests/test_rotation.cu
    ${CORE_SOURCES}
    src/filters/rotation.cu
)
target_link_libraries(test_rotation PRIVATE stdc++ m cudart)

add_executable(test_blur
    tests/test_blur.cu
    ${CORE_SOURCES}
    src/filters/blur.cu
)
target_link_libraries(test_blur PRIVATE stdc++ m cudart)

add_executable(test_parallel
    tests/test_parallel.cu
    ${CORE_SOURCES}
)
target_link_libraries(test_parallel PRIVATE stdc++ m cudart)

