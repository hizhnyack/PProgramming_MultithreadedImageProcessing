#include <stdio.h>
#include <string>
#include "utils/image_loader.h"
#include "utils/batch_processor.h"
#include "filters/grayscale.h"
#include "filters/rotation.h"
#include "filters/blur.h"

void printUsage(const char* program_name) {
    printf("Usage: %s <command> [options]\n\n", program_name);
    printf("Commands:\n");
    printf("  grayscale <input> <output>          - Convert to grayscale\n");
    printf("  rotate90 <input> <output>           - Rotate 90 degrees\n");
    printf("  rotate180 <input> <output>          - Rotate 180 degrees\n");
    printf("  rotate270 <input> <output>          - Rotate 270 degrees\n");
    printf("  blur <input> <output> <radius>      - Apply blur filter\n");
    printf("  batch <filter> <input_dir> <output_dir> - Process directory\n");
    printf("\nExample:\n");
    printf("  %s grayscale input.png output.png\n", program_name);
    printf("  %s batch grayscale ./images ./output\n", program_name);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string command = argv[1];
    
    // Single file processing
    if (command == "grayscale" && argc == 4) {
        ImageData image;
        if (!ImageLoader::load(argv[2], image)) {
            return 1;
        }
        
        if (!GrayscaleFilter::applyInPlace(image)) {
            fprintf(stderr, "Failed to apply grayscale filter\n");
            return 1;
        }
        
        cudaMemcpy(image.data, image.gpu_data, image.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], image)) {
            return 1;
        }
        
        printf("✓ Grayscale filter applied successfully\n");
        return 0;
    }
    
    else if (command == "rotate90" && argc == 4) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        if (!RotationFilter::rotate90(input, output)) {
            fprintf(stderr, "Failed to rotate image\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Image rotated 90° successfully\n");
        return 0;
    }
    
    else if (command == "rotate180" && argc == 4) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        if (!RotationFilter::rotate180(input, output)) {
            fprintf(stderr, "Failed to rotate image\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Image rotated 180° successfully\n");
        return 0;
    }
    
    else if (command == "rotate270" && argc == 4) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        if (!RotationFilter::rotate270(input, output)) {
            fprintf(stderr, "Failed to rotate image\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Image rotated 270° successfully\n");
        return 0;
    }
    
    else if (command == "blur" && argc == 5) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        int radius = atoi(argv[4]);
        if (radius < 1 || radius > 50) {
            fprintf(stderr, "Blur radius must be between 1 and 50\n");
            return 1;
        }
        
        if (!BlurFilter::applyBox(input, output, radius)) {
            fprintf(stderr, "Failed to apply blur filter\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Blur filter applied successfully\n");
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
    
    else {
        fprintf(stderr, "Invalid command or arguments\n\n");
        printUsage(argv[0]);
        return 1;
    }
}
