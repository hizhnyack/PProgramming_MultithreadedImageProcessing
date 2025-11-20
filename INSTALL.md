# Установка и использование библиотеки

## Быстрый старт

### 1. Сборка библиотеки

```bash
# Клонируем репозиторий (если еще не клонирован)
git clone https://github.com/hizhnyack/PProgramming_MultithreadedImageProcessing.git
cd PProgramming_MultithreadedImageProcessing

# Собираем библиотеку
mkdir -p build
cd build
cmake ..
make -j4
```

После сборки получаем:
- `build/libimage_processing.a` - статическая библиотека
- `build/image_processor` - готовая программа для обработки изображений
- `build/test_*` - тестовые программы

### 2. Использование готовой программы

Самый простой способ - использовать готовую программу `image_processor`:

```bash
cd build

# Преобразование в оттенки серого
./image_processor grayscale input.png output.png

# Поворот на 90 градусов
./image_processor rotate90 input.png output.png

# Размытие
./image_processor blur input.png output.png 5

# Пакетная обработка всех изображений в папке
./image_processor batch grayscale ../foto ../output
```

### 3. Использование библиотеки в своем проекте

#### Вариант A: Простая компиляция

Создай файл `my_program.cu`:

```cpp
#include "filters/grayscale.h"
#include "utils/image_loader.h"
#include <stdio.h>

int main() {
    // Загрузка изображения
    ImageData image;
    if (!ImageLoader::load("input.png", image)) {
        printf("Ошибка загрузки изображения\n");
        return 1;
    }
    
    printf("Изображение загружено: %dx%d\n", image.width, image.height);
    
    // Применение фильтра grayscale
    if (!GrayscaleFilter::applyInPlace(image)) {
        printf("Ошибка применения фильтра\n");
        return 1;
    }
    
    // Копирование результата с GPU на CPU
    cudaMemcpy(image.data, image.gpu_data, 
               image.size_bytes, cudaMemcpyDeviceToHost);
    
    // Сохранение результата
    if (!ImageLoader::save("output.png", image)) {
        printf("Ошибка сохранения изображения\n");
        return 1;
    }
    
    printf("Готово! Результат сохранен в output.png\n");
    
    return 0;
}
```

Компиляция:

```bash
# Путь к библиотеке
LIB_PATH=/home/anton/PProgramming_MultithreadedImageProcessing

# Компиляция
nvcc my_program.cu \
    -I${LIB_PATH}/src \
    -L${LIB_PATH}/build \
    -limage_processing \
    -lcudart \
    -o my_program

# Запуск
./my_program
```

#### Вариант B: Использование CMake

Создай `CMakeLists.txt` для своего проекта:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 86)

# Путь к библиотеке
set(IMAGE_LIB_PATH "/home/anton/PProgramming_MultithreadedImageProcessing")

# Подключаем заголовочные файлы
include_directories(${IMAGE_LIB_PATH}/src)

# Подключаем библиотеку
link_directories(${IMAGE_LIB_PATH}/build)

# Создаем исполняемый файл
add_executable(my_program my_program.cu)

# Линкуем с библиотекой
target_link_libraries(my_program PRIVATE
    image_processing
    cudart
    stdc++
    m
)
```

Сборка:

```bash
mkdir build && cd build
cmake ..
make
./my_program
```

## Примеры использования

### Пример 1: Простая обработка одного изображения

```cpp
#include "filters/grayscale.h"
#include "utils/image_loader.h"

int main() {
    ImageData image;
    
    // Загрузка
    ImageLoader::load("photo.jpg", image);
    
    // Обработка
    GrayscaleFilter::applyInPlace(image);
    
    // Копирование с GPU
    cudaMemcpy(image.data, image.gpu_data, 
               image.size_bytes, cudaMemcpyDeviceToHost);
    
    // Сохранение
    ImageLoader::save("photo_gray.jpg", image);
    
    return 0;
}
```

### Пример 2: Поворот изображения

```cpp
#include "filters/rotation.h"
#include "utils/image_loader.h"

int main() {
    ImageData input, output;
    
    ImageLoader::load("photo.jpg", input);
    
    // Поворот на 90 градусов
    RotationFilter::rotate90(input, output);
    
    cudaMemcpy(output.data, output.gpu_data, 
               output.size_bytes, cudaMemcpyDeviceToHost);
    
    ImageLoader::save("photo_rotated.jpg", output);
    
    return 0;
}
```

### Пример 3: Размытие

```cpp
#include "filters/blur.h"
#include "utils/image_loader.h"

int main() {
    ImageData image;
    
    ImageLoader::load("photo.jpg", image);
    
    // Gaussian Blur с радиусом 5
    BlurFilter::applyGaussianBlur(image, 5, 2.0f);
    
    cudaMemcpy(image.data, image.gpu_data, 
               image.size_bytes, cudaMemcpyDeviceToHost);
    
    ImageLoader::save("photo_blurred.jpg", image);
    
    return 0;
}
```

### Пример 4: Параллельная обработка нескольких изображений

```cpp
#include "core/parallel_processor.h"
#include "utils/image_loader.h"
#include <vector>

int main() {
    // Инициализация
    ParallelProcessor::ParallelConfig config;
    config.maxConcurrentStreams = 4;
    config.verbose = true;
    
    ParallelProcessor::initialize(config);
    
    // Загрузка изображений
    std::vector<std::vector<unsigned char>> inputImages;
    std::vector<std::string> files = {"img1.jpg", "img2.jpg", "img3.jpg"};
    
    for (const auto& file : files) {
        ImageData img;
        ImageLoader::load(file, img);
        
        std::vector<unsigned char> data(img.size_bytes);
        memcpy(data.data(), img.data, img.size_bytes);
        inputImages.push_back(data);
    }
    
    // Параллельная обработка
    std::vector<std::vector<unsigned char>> outputImages;
    auto stats = ParallelProcessor::processBatchParallel(
        inputImages,
        outputImages,
        1920, 1080,  // размеры изображений
        "grayscale"
    );
    
    // Вывод статистики
    stats.print();
    
    // Сохранение результатов
    for (size_t i = 0; i < outputImages.size(); i++) {
        ImageData img;
        img.width = 1920;
        img.height = 1080;
        img.channels = 3;
        img.data = outputImages[i].data();
        
        ImageLoader::save("output_" + std::to_string(i) + ".jpg", img);
    }
    
    ParallelProcessor::cleanup();
    
    return 0;
}
```

## Доступные фильтры и методы

### GrayscaleFilter

```cpp
// Применить на месте (изменяет исходное изображение)
GrayscaleFilter::applyInPlace(image);

// Создать новое изображение с результатом
GrayscaleFilter::apply(input, output);

// Выбор метода преобразования
GrayscaleFilter::applyWithMethod(image, GrayscaleMethod::LUMINOSITY);
GrayscaleFilter::applyWithMethod(image, GrayscaleMethod::AVERAGE);
GrayscaleFilter::applyWithMethod(image, GrayscaleMethod::LIGHTNESS);
```

### RotationFilter

```cpp
// Фиксированные углы
RotationFilter::rotate90(input, output);
RotationFilter::rotate180(input, output);
RotationFilter::rotate270(input, output);

// Произвольный угол (в градусах)
RotationFilter::rotateArbitrary(input, output, 45.0f);
```

### BlurFilter

```cpp
// Box Blur
BlurFilter::applyBoxBlur(image, radius);

// Gaussian Blur
BlurFilter::applyGaussianBlur(image, radius, sigma);

// Motion Blur
BlurFilter::applyMotionBlur(image, length, angle);
```

### ParallelProcessor

```cpp
// Инициализация
ParallelProcessor::ParallelConfig config;
config.maxConcurrentStreams = 4;  // Количество параллельных потоков
config.blockSize = 16;            // Размер CUDA блока
config.verbose = true;            // Подробный вывод

ParallelProcessor::initialize(config);

// Пакетная обработка
auto stats = ParallelProcessor::processBatchParallel(
    inputImages,   // std::vector<std::vector<unsigned char>>
    outputImages,  // std::vector<std::vector<unsigned char>>
    width,         // int
    height,        // int
    "grayscale"    // тип фильтра: "grayscale" или "invert"
);

// Вывод статистики
stats.print();

// Очистка
ParallelProcessor::cleanup();
```

## Веб-интерфейс

Библиотека также доступна через веб-интерфейс:

```bash
cd web
./start_server.sh

# Открой в браузере: http://127.0.0.1:5000
```

Подробнее: [web/README_WEB.md](web/README_WEB.md)

## Тестирование

```bash
cd build

# Тест оттенков серого
./test_grayscale

# Тест поворота
./test_rotation

# Тест размытия
./test_blur

# Тест параллельной обработки (8 изображений)
./test_parallel 8

# Тест с большим количеством изображений
./test_parallel 64
```

## Требования

- **CUDA Toolkit** 11.0 или выше
- **CMake** 3.18 или выше
- **GPU** с compute capability 8.6+ (Ampere или новее)
- **Компилятор:** GCC 9+ (Linux) или MSVC 2019+ (Windows)

## Проверка установки CUDA

```bash
# Проверка версии CUDA
nvcc --version

# Проверка доступных GPU
nvidia-smi

# Информация о GPU
./build/test_parallel 1
```

## Устранение проблем

### Ошибка: "libimage_processing.a: No such file"

```bash
# Убедись что библиотека собрана
cd PProgramming_MultithreadedImageProcessing/build
ls -lh libimage_processing.a

# Если файла нет - пересобери
cmake ..
make -j4
```

### Ошибка: "undefined reference to cudaMalloc"

Добавь `-lcudart` при компиляции:

```bash
nvcc my_program.cu -limage_processing -lcudart -o my_program
```

### Ошибка: "CUDA error: no CUDA-capable device"

Проверь, что GPU доступен:

```bash
nvidia-smi
```

## Документация

- [README.md](README.md) - Общее описание проекта
- [LIBRARY.md](LIBRARY.md) - Подробная документация библиотеки
- [QUICK_START.md](QUICK_START.md) - Быстрый старт
- [web/README_WEB.md](web/README_WEB.md) - Веб-интерфейс

## Поддержка

При возникновении проблем:
1. Проверь, что CUDA установлена: `nvcc --version`
2. Проверь, что GPU доступен: `nvidia-smi`
3. Запусти тесты: `./build/test_parallel 8`
4. Посмотри логи компиляции

---

**Готово!** Теперь ты можешь использовать библиотеку в своих проектах! 

