#include "cpu_filters.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace CPUFilters {

// Grayscale - многопоточная версия
bool grayscale(const ImageData& input, ImageData& output) {
    if (!input.data || input.channels < 1) {
        fprintf(stderr, "[CPU] Invalid input image for grayscale filter\n");
        return false;
    }
    
    // Подготовка выходного изображения
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = 1;
        output.size_bytes = input.width * input.height * sizeof(unsigned char);
        output.data = new unsigned char[output.size_bytes];
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int rows_per_thread = input.height / num_threads;
    
    auto worker = [&](int start_row, int end_row) {
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < input.width; x++) {
                int idx = (y * input.width + x) * input.channels;
                
                if (input.channels >= 3) {
                    unsigned char r = input.data[idx];
                    unsigned char g = input.data[idx + 1];
                    unsigned char b = input.data[idx + 2];
                    
                    // 0.299*R + 0.587*G + 0.114*B
                    unsigned char gray = (77 * r + 150 * g + 29 * b) / 256;
                    output.data[y * input.width + x] = gray;
                } else if (input.channels == 1) {
                    output.data[y * input.width + x] = input.data[idx];
                }
            }
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? input.height : (i + 1) * rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return true;
}

// Rotate 90
bool rotate90(const ImageData& input, ImageData& output) {
    if (!input.data || input.channels < 1) {
        fprintf(stderr, "[CPU] Invalid input image for rotation filter\n");
        return false;
    }
    
    int output_width = input.height;
    int output_height = input.width;
    
    if (!output.data) {
        output.width = output_width;
        output.height = output_height;
        output.channels = input.channels;
        output.size_bytes = output.width * output.height * output.channels * sizeof(unsigned char);
        output.data = new unsigned char[output.size_bytes];
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int rows_per_thread = output_height / num_threads;
    
    auto worker = [&](int start_row, int end_row) {
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < output_width; x++) {
                int in_x = y;
                int in_y = output_width - 1 - x;
                
                int in_idx = (in_y * input.width + in_x) * input.channels;
                int out_idx = (y * output_width + x) * input.channels;
                
                for (int c = 0; c < input.channels; c++) {
                    output.data[out_idx + c] = input.data[in_idx + c];
                }
            }
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? output_height : (i + 1) * rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return true;
}

// Rotate 180
bool rotate180(const ImageData& input, ImageData& output) {
    if (!input.data || input.channels < 1) {
        fprintf(stderr, "[CPU] Invalid input image for rotation filter\n");
        return false;
    }
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.size_bytes = input.size_bytes;
        output.data = new unsigned char[output.size_bytes];
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int rows_per_thread = input.height / num_threads;
    
    auto worker = [&](int start_row, int end_row) {
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < input.width; x++) {
                int new_x = input.width - 1 - x;
                int new_y = input.height - 1 - y;
                
                int in_idx = (y * input.width + x) * input.channels;
                int out_idx = (new_y * input.width + new_x) * input.channels;
                
                for (int c = 0; c < input.channels; c++) {
                    output.data[out_idx + c] = input.data[in_idx + c];
                }
            }
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? input.height : (i + 1) * rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return true;
}

// Rotate 270
bool rotate270(const ImageData& input, ImageData& output) {
    if (!input.data || input.channels < 1) {
        fprintf(stderr, "[CPU] Invalid input image for rotation filter\n");
        return false;
    }
    
    int output_width = input.height;
    int output_height = input.width;
    
    if (!output.data) {
        output.width = output_width;
        output.height = output_height;
        output.channels = input.channels;
        output.size_bytes = output.width * output.height * output.channels * sizeof(unsigned char);
        output.data = new unsigned char[output.size_bytes];
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int rows_per_thread = output_height / num_threads;
    
    auto worker = [&](int start_row, int end_row) {
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < output_width; x++) {
                int in_x = output_height - 1 - y;
                int in_y = x;
                
                int in_idx = (in_y * input.width + in_x) * input.channels;
                int out_idx = (y * output_width + x) * input.channels;
                
                for (int c = 0; c < input.channels; c++) {
                    output.data[out_idx + c] = input.data[in_idx + c];
                }
            }
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? output_height : (i + 1) * rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return true;
}

// Rotate Arbitrary
bool rotateArbitrary(const ImageData& input, ImageData& output, 
                    float angle, unsigned char background) {
    if (!input.data || input.channels < 1) {
        fprintf(stderr, "[CPU] Invalid input image for rotation filter\n");
        return false;
    }
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.size_bytes = input.size_bytes;
        output.data = new unsigned char[output.size_bytes];
    }
    
    float angle_rad = angle * M_PI / 180.0f;
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);
    
    float cx = input.width / 2.0f;
    float cy = input.height / 2.0f;
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int rows_per_thread = input.height / num_threads;
    
    auto worker = [&](int start_row, int end_row) {
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < input.width; x++) {
                float dx = x - cx;
                float dy = y - cy;
                
                float src_x = dx * cos_angle + dy * sin_angle + cx;
                float src_y = -dx * sin_angle + dy * cos_angle + cy;
                
                int out_idx = (y * input.width + x) * input.channels;
                
                if (src_x >= 0 && src_x < input.width - 1 && 
                    src_y >= 0 && src_y < input.height - 1) {
                    int x0 = (int)src_x;
                    int y0 = (int)src_y;
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    
                    float fx = src_x - x0;
                    float fy = src_y - y0;
                    
                    for (int c = 0; c < input.channels; c++) {
                        float v00 = input.data[(y0 * input.width + x0) * input.channels + c];
                        float v10 = input.data[(y0 * input.width + x1) * input.channels + c];
                        float v01 = input.data[(y1 * input.width + x0) * input.channels + c];
                        float v11 = input.data[(y1 * input.width + x1) * input.channels + c];
                        
                        float v0 = v00 * (1 - fx) + v10 * fx;
                        float v1 = v01 * (1 - fx) + v11 * fx;
                        float v = v0 * (1 - fy) + v1 * fy;
                        
                        output.data[out_idx + c] = (unsigned char)v;
                    }
                } else {
                    for (int c = 0; c < input.channels; c++) {
                        output.data[out_idx + c] = background;
                    }
                }
            }
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? input.height : (i + 1) * rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return true;
}

// Box Blur
bool blurBox(const ImageData& input, ImageData& output, int radius) {
    if (!input.data || input.channels < 1) {
        fprintf(stderr, "[CPU] Invalid input image for blur filter\n");
        return false;
    }
    
    if (!output.data) {
        output.width = input.width;
        output.height = input.height;
        output.channels = input.channels;
        output.size_bytes = input.size_bytes;
        output.data = new unsigned char[output.size_bytes];
    }
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    int rows_per_thread = input.height / num_threads;
    
    auto worker = [&](int start_row, int end_row) {
        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < input.width; x++) {
                for (int c = 0; c < input.channels; c++) {
                    float sum = 0.0f;
                    int count = 0;
                    
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int nx = x + dx;
                            int ny = y + dy;
                            
                            if (nx >= 0 && nx < input.width && 
                                ny >= 0 && ny < input.height) {
                                int idx = (ny * input.width + nx) * input.channels + c;
                                sum += input.data[idx];
                                count++;
                            }
                        }
                    }
                    
                    int out_idx = (y * input.width + x) * input.channels + c;
                    output.data[out_idx] = (unsigned char)(sum / count);
                }
            }
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? input.height : (i + 1) * rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return true;
}

} // namespace CPUFilters

