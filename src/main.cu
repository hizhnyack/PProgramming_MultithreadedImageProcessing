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
    printf("  grayscale <input> <output>                    - Convert to grayscale (standard)\n");
    printf("  grayscale_weighted <input> <output> <R> <G> <B> - Weighted grayscale (custom)\n");
    printf("  rotate90 <input> <output>                     - Rotate 90 degrees\n");
    printf("  rotate180 <input> <output>                    - Rotate 180 degrees\n");
    printf("  rotate270 <input> <output>                    - Rotate 270 degrees\n");
    printf("  blur <input> <output> <radius>                - Box blur (fast)\n");
    printf("  blur_gaussian <input> <output> <sigma>        - Gaussian blur (quality)\n");
    printf("  blur_separable <input> <output> <sigma>       - Separable Gaussian (optimized)\n");
    printf("  blur_motion <input> <output> <length> <angle> - Motion blur (effect)\n");
    printf("  batch <filter> <input_dir> <output_dir>       - Process directory (multithreaded)\n");
    printf("  batch-pipeline <filter> <input_dir> <output_dir> - Process directory (pipeline - FASTER!)\n");
    printf("\nExamples:\n");
    printf("  %s grayscale input.png output.png\n", program_name);
    printf("  %s grayscale_weighted input.png output.png 0.5 0.3 0.2\n", program_name);
    printf("  %s rotate90 input.png output.png\n", program_name);
    printf("  %s rotate_arbitrary input.png output.png 45.5\n", program_name);
    printf("  %s blur input.png output.png 10\n", program_name);
    printf("  %s blur_gaussian input.png output.png 2.5\n", program_name);
    printf("  %s blur_motion input.png output.png 20 45\n", program_name);
    printf("  %s batch-pipeline grayscale ./images ./output\n", program_name);
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
    
    else if (command == "grayscale_weighted" && argc == 7) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        float r_weight = atof(argv[4]);
        float g_weight = atof(argv[5]);
        float b_weight = atof(argv[6]);
        
        // Проверка весов
        if (r_weight < 0 || r_weight > 1 || 
            g_weight < 0 || g_weight > 1 || 
            b_weight < 0 || b_weight > 1) {
            fprintf(stderr, "Weights must be between 0 and 1\n");
            return 1;
        }
        
        if (!GrayscaleFilter::applyWithWeights(input, output, r_weight, g_weight, b_weight)) {
            fprintf(stderr, "Failed to apply weighted grayscale filter\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Weighted grayscale applied (R=%.3f, G=%.3f, B=%.3f)\n", 
               r_weight, g_weight, b_weight);
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
    
    else if (command == "rotate_arbitrary" && argc == 5) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        float angle = atof(argv[4]);
        if (angle < 0 || angle > 360) {
            fprintf(stderr, "Angle must be between 0 and 360 degrees\n");
            return 1;
        }
        
        if (!RotationFilter::rotateArbitrary(input, output, angle)) {
            fprintf(stderr, "Failed to rotate image\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Image rotated %.1f° successfully\n", angle);
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
    
    else if (command == "blur_gaussian" && argc == 5) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        float sigma = atof(argv[4]);
        if (sigma < 0.1 || sigma > 20.0) {
            fprintf(stderr, "Gaussian sigma must be between 0.1 and 20.0\n");
            return 1;
        }
        
        if (!BlurFilter::applyGaussian(input, output, sigma)) {
            fprintf(stderr, "Failed to apply Gaussian blur\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Gaussian blur applied successfully (sigma=%.2f)\n", sigma);
        return 0;
    }
    
    else if (command == "blur_separable" && argc == 5) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        float sigma = atof(argv[4]);
        if (sigma < 0.1 || sigma > 20.0) {
            fprintf(stderr, "Gaussian sigma must be between 0.1 and 20.0\n");
            return 1;
        }
        
        if (!BlurFilter::applyGaussianSeparable(input, output, sigma)) {
            fprintf(stderr, "Failed to apply Separable Gaussian blur\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Separable Gaussian blur applied successfully (sigma=%.2f)\n", sigma);
        return 0;
    }
    
    else if (command == "blur_motion" && argc == 6) {
        ImageData input, output;
        if (!ImageLoader::load(argv[2], input)) {
            return 1;
        }
        
        int length = atoi(argv[4]);
        float angle = atof(argv[5]);
        
        if (length < 1 || length > 100) {
            fprintf(stderr, "Motion blur length must be between 1 and 100\n");
            return 1;
        }
        
        if (!BlurFilter::applyMotion(input, output, length, angle)) {
            fprintf(stderr, "Failed to apply Motion blur\n");
            return 1;
        }
        
        cudaMemcpy(output.data, output.gpu_data, output.size_bytes, cudaMemcpyDeviceToHost);
        
        if (!ImageLoader::save(argv[3], output)) {
            return 1;
        }
        
        printf("✓ Motion blur applied successfully (length=%d, angle=%.1f°)\n", length, angle);
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
