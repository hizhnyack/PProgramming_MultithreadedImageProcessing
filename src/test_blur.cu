#include <iostream>
#include "core/image_processor.h"
#include "filters/blur.h"

void createTestImage(ImageData& img, int width, int height, int channels) {
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.size_bytes = width * height * channels * sizeof(unsigned char);
    img.data = new unsigned char[img.size_bytes];
    
    for (size_t i = 0; i < img.size_bytes; i++) {
        img.data[i] = (i % 256);
    }
}

int main() {
    std::cout << "=== Test Blur Filter ===" << std::endl;
    
    ImageData input, output;
    createTestImage(input, 256, 256, 3);
    
    ImageProcessor processor;
    
    if (!processor.allocateGPUMemory(input)) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        return 1;
    }
    
    if (!processor.copyHostToDevice(input, input)) {
        std::cerr << "Failed to copy to device" << std::endl;
        return 1;
    }
    
    std::cout << "Testing box blur..." << std::endl;
    if (!BlurFilter::applyBox(input, output, 3)) {
        std::cerr << "Box blur failed" << std::endl;
        return 1;
    }
    
    if (!processor.copyDeviceToHost(output, output)) {
        std::cerr << "Failed to copy from device" << std::endl;
        return 1;
    }
    
    std::cout << "✓ Blur filter passed" << std::endl;
    std::cout << "Input: " << input.width << "x" << input.height << "x" << input.channels << std::endl;
    std::cout << "Output: " << output.width << "x" << output.height << "x" << output.channels << std::endl;
    
    return 0;
}

