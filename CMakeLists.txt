cmake_minimum_required(VERSION 3.18)
project(MultithreadedImageProcessing CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Обход проверки версии Visual Studio
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
)

set(FILTER_SOURCES
    src/filters/grayscale.cu
    src/filters/rotation.cu
    src/filters/blur.cu
)

add_executable(image_processor
    src/main.cu
    ${CORE_SOURCES}
    ${FILTER_SOURCES}
)

set_target_properties(image_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(image_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler -Wall
    >
)

