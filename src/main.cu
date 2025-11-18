#include <iostream>
#include <cstring>
#include "core/image_processor.h"
#include "filters/grayscale.h"
#include "filters/rotation.h"
#include "filters/blur.h"

void createTestImage(ImageData& img, int width, int height, int channels) {
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.size_bytes = width * height * channels * sizeof(unsigned char);
    img.data = new unsigned char[img.size_bytes];
    
    // Создаём простой градиент
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            if (channels >= 3) {
                img.data[idx] = (unsigned char)(255 * x / width);     // R
                img.data[idx + 1] = (unsigned char)(255 * y / height); // G
                img.data[idx + 2] = 128;                               // B
            } else {
                img.data[idx] = (unsigned char)(255 * x / width);
            }
        }
    }
}

void printImageStats(const ImageData& img, const char* name) {
    if (!img.data || img.size_bytes == 0) {
        std::cout << name << ": Empty image" << std::endl;
        return;
    }
    
    long sum = 0;
    for (size_t i = 0; i < img.size_bytes; i++) {
        sum += img.data[i];
    }
    
    std::cout << name << ": " 
              << img.width << "x" << img.height 
              << "x" << img.channels 
              << ", avg=" << (sum / img.size_bytes)
              << std::endl;
}

bool testGrayscale() {
    std::cout << "\n=== Test Grayscale Filter ===" << std::endl;
    
    ImageData input, output;
    createTestImage(input, 256, 256, 3);
    
    ImageProcessor processor;
    if (!processor.allocateGPUMemory(input)) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        return false;
    }
    
    if (!processor.copyHostToDevice(input, input)) {
        std::cerr << "Failed to copy to device" << std::endl;
        return false;
    }
    
    if (!GrayscaleFilter::apply(input, output)) {
        std::cerr << "Grayscale filter failed" << std::endl;
        return false;
    }
    
    if (!processor.copyDeviceToHost(output, output)) {
        std::cerr << "Failed to copy from device" << std::endl;
        return false;
    }
    
    printImageStats(input, "Input");
    printImageStats(output, "Output");
    
    std::cout << "✓ Grayscale filter passed" << std::endl;
    return true;
}

bool testRotation() {
    std::cout << "\n=== Test Rotation Filter ===" << std::endl;
    
    ImageData input, output90, output180, output270;
    createTestImage(input, 128, 128, 3);
    
    ImageProcessor processor;
    processor.allocateGPUMemory(input);
    processor.copyHostToDevice(input, input);
    
    // Test rotate90
    if (!RotationFilter::rotate90(input, output90)) {
        std::cerr << "Rotate90 failed" << std::endl;
        return false;
    }
    processor.copyDeviceToHost(output90, output90);
    
    // Test rotate180
    if (!RotationFilter::rotate180(input, output180)) {
        std::cerr << "Rotate180 failed" << std::endl;
        return false;
    }
    processor.copyDeviceToHost(output180, output180);
    
    // Test rotate270
    if (!RotationFilter::rotate270(input, output270)) {
        std::cerr << "Rotate270 failed" << std::endl;
        return false;
    }
    processor.copyDeviceToHost(output270, output270);
    
    printImageStats(input, "Input");
    printImageStats(output90, "Rotated 90°");
    printImageStats(output180, "Rotated 180°");
    printImageStats(output270, "Rotated 270°");
    
    // Проверка размеров
    if (output90.width != input.height || output90.height != input.width) {
        std::cerr << "Rotate90 dimensions incorrect" << std::endl;
        return false;
    }
    
    std::cout << "✓ Rotation filter passed" << std::endl;
    return true;
}

bool testBlur() {
    std::cout << "\n=== Test Blur Filter ===" << std::endl;
    
    ImageData input, outputBox, outputGaussian;
    createTestImage(input, 128, 128, 3);
    
    ImageProcessor processor;
    processor.allocateGPUMemory(input);
    processor.copyHostToDevice(input, input);
    
    // Test Box Blur
    if (!BlurFilter::applyBox(input, outputBox, 3)) {
        std::cerr << "Box blur failed" << std::endl;
        return false;
    }
    processor.copyDeviceToHost(outputBox, outputBox);
    
    // Test Gaussian Blur
    if (!BlurFilter::applyGaussian(input, outputGaussian, 2.0f)) {
        std::cerr << "Gaussian blur failed" << std::endl;
        return false;
    }
    processor.copyDeviceToHost(outputGaussian, outputGaussian);
    
    printImageStats(input, "Input");
    printImageStats(outputBox, "Box Blur");
    printImageStats(outputGaussian, "Gaussian Blur");
    
    std::cout << "✓ Blur filter passed" << std::endl;
    return true;
}

bool testCombined() {
    std::cout << "\n=== Test Combined Filters ===" << std::endl;
    
    ImageData input, blurred, rotated, grayscale;
    createTestImage(input, 128, 128, 3);
    
    ImageProcessor processor;
    processor.allocateGPUMemory(input);
    processor.copyHostToDevice(input, input);
    
    // Blur -> Rotate -> Grayscale
    if (!BlurFilter::applyGaussian(input, blurred, 1.5f)) {
        std::cerr << "Blur failed" << std::endl;
        return false;
    }
    
    if (!RotationFilter::rotate90(blurred, rotated)) {
        std::cerr << "Rotation failed" << std::endl;
        return false;
    }
    
    if (!GrayscaleFilter::apply(rotated, grayscale)) {
        std::cerr << "Grayscale failed" << std::endl;
        return false;
    }
    
    processor.copyDeviceToHost(grayscale, grayscale);
    
    printImageStats(input, "Input");
    printImageStats(grayscale, "Final (Blur->Rotate->Gray)");
    
    std::cout << "✓ Combined filters passed" << std::endl;
    return true;
}

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "Image Processing Tests" << std::endl;
    std::cout << "==================================" << std::endl;
    
    try {
        ImageProcessor processor;
        std::cout << "CUDA devices found: " << processor.getDeviceCount() << std::endl;
        
        bool allPassed = true;
        
        allPassed &= testGrayscale();
        allPassed &= testRotation();
        allPassed &= testBlur();
        allPassed &= testCombined();
        
        std::cout << "\n==================================" << std::endl;
        if (allPassed) {
            std::cout << "✓ All tests PASSED!" << std::endl;
        } else {
            std::cout << "✗ Some tests FAILED!" << std::endl;
        }
        std::cout << "==================================" << std::endl;
        
        return allPassed ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

