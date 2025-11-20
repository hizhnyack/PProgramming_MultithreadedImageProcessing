# Библиотека для обработки изображений на GPU

## Описание

`libimage_processing.a` - статическая библиотека для высокопроизводительной обработки изображений с использованием CUDA и параллельных вычислений на GPU.

## Возможности

### Фильтры обработки изображений

1. **Grayscale (Оттенки серого)**
   - Несколько алгоритмов: Lightness, Average, Luminosity
   - Поддержка различных формул преобразования

2. **Rotation (Поворот)**
   - Фиксированные углы: 90°, 180°, 270°
   - Произвольный угол поворота
   - Различные типы размытия: Box, Gaussian, Motion

3. **Blur (Размытие)**
   - Box Blur
   - Gaussian Blur
   - Motion Blur

### Параллельная обработка

- **ParallelProcessor** - система для пакетной обработки нескольких изображений одновременно
- Использование CUDA streams для асинхронной обработки
- Ускорение до **2x** при обработке больших пакетов изображений

## Структура библиотеки

```
libimage_processing.a
├── Core (Ядро)
│   ├── image_processor.cu     - Базовая обработка изображений
│   ├── cuda_kernels.cu        - CUDA ядра
│   └── parallel_processor.cu  - Параллельная обработка
├── Filters (Фильтры)
│   ├── grayscale.cu           - Преобразование в оттенки серого
│   ├── rotation.cu            - Поворот изображений
│   └── blur.cu                - Размытие
└── Utils (Утилиты)
    ├── image_loader.cu        - Загрузка/сохранение изображений
    └── batch_processor.cu     - Пакетная обработка
```

## Использование библиотеки

### Подключение в проект

#### CMake

```cmake
# Добавьте путь к библиотеке
link_directories(/path/to/PProgramming_MultithreadedImageProcessing/build)

# Подключите заголовочные файлы
include_directories(/path/to/PProgramming_MultithreadedImageProcessing/src)

# Линкуйте библиотеку
target_link_libraries(your_project PRIVATE
    image_processing
    cudart
    stdc++
    m
)
```

#### Компиляция вручную

```bash
nvcc your_program.cu \
    -L/path/to/build \
    -limage_processing \
    -lcudart \
    -I/path/to/src \
    -o your_program
```

### Пример использования

#### Простая обработка одного изображения

```cpp
#include "filters/grayscale.h"
#include "utils/image_loader.h"

int main() {
    // Загрузка изображения
    ImageData image;
    if (!ImageLoader::load("input.png", image)) {
        return 1;
    }
    
    // Применение фильтра
    if (!GrayscaleFilter::applyInPlace(image)) {
        return 1;
    }
    
    // Копирование результата с GPU
    cudaMemcpy(image.data, image.gpu_data, 
               image.size_bytes, cudaMemcpyDeviceToHost);
    
    // Сохранение результата
    ImageLoader::save("output.png", image);
    
    return 0;
}
```

#### Параллельная обработка нескольких изображений

```cpp
#include "core/parallel_processor.h"
#include <vector>

int main() {
    // Инициализация
    ParallelProcessor::ParallelConfig config;
    config.maxConcurrentStreams = 4;
    config.verbose = true;
    
    if (!ParallelProcessor::initialize(config)) {
        return 1;
    }
    
    // Подготовка данных
    std::vector<std::vector<unsigned char>> inputImages;
    std::vector<std::vector<unsigned char>> outputImages;
    
    // ... загрузка изображений в inputImages ...
    
    // Параллельная обработка
    auto stats = ParallelProcessor::processBatchParallel(
        inputImages,
        outputImages,
        width,
        height,
        "grayscale"
    );
    
    // Вывод статистики
    stats.print();
    
    // Очистка
    ParallelProcessor::cleanup();
    
    return 0;
}
```

## API Документация

### ParallelProcessor

#### Конфигурация

```cpp
struct ParallelConfig {
    int maxConcurrentStreams = 4;  // Количество параллельных потоков
    int blockSize = 16;            // Размер CUDA блока
    bool enableTiming = true;      // Замер времени
    bool verbose = false;          // Подробный вывод
};
```

#### Методы

- `initialize(config)` - Инициализация процессора
- `processBatchParallel(...)` - Пакетная обработка изображений
- `printGPUInfo()` - Информация о GPU
- `cleanup()` - Освобождение ресурсов

### GrayscaleFilter

```cpp
// Применение фильтра на месте
bool applyInPlace(ImageData& image);

// Применение с сохранением оригинала
bool apply(const ImageData& input, ImageData& output);

// Выбор алгоритма
bool applyWithMethod(ImageData& image, GrayscaleMethod method);
```

### RotationFilter

```cpp
// Фиксированный поворот
bool rotate90(const ImageData& input, ImageData& output);
bool rotate180(const ImageData& input, ImageData& output);
bool rotate270(const ImageData& input, ImageData& output);

// Произвольный угол
bool rotateArbitrary(const ImageData& input, ImageData& output, float angle);
```

### BlurFilter

```cpp
// Box Blur
bool applyBoxBlur(ImageData& image, int radius);

// Gaussian Blur
bool applyGaussianBlur(ImageData& image, int radius, float sigma);

// Motion Blur
bool applyMotionBlur(ImageData& image, int length, float angle);
```

## Производительность

### Тесты параллельной обработки (RTX 4060)

| Количество изображений | Последовательно | Параллельно | Ускорение |
|------------------------|-----------------|-------------|-----------|
| 8                      | 248 ms          | 321 ms      | 0.77x     |
| 16                     | 494 ms          | 399 ms      | 1.24x     |
| 32                     | 987 ms          | 608 ms      | 1.62x     |
| 64                     | 1936 ms         | 927 ms      | 2.09x     |

**Вывод:** Параллельная обработка эффективна при количестве изображений > 10.

## Требования

- **CUDA Toolkit** 11.0+
- **CMake** 3.18+
- **GPU** с compute capability 8.6+ (Ampere или новее)
- **Компилятор:** GCC 9+ или MSVC 2019+

## Сборка библиотеки

```bash
# Клонирование репозитория
git clone https://github.com/hizhnyack/PProgramming_MultithreadedImageProcessing.git
cd PProgramming_MultithreadedImageProcessing

# Сборка
mkdir build && cd build
cmake ..
make -j4

# Результат: build/libimage_processing.a
```

## Тестирование

```bash
# Запуск всех тестов
cd build
./test_grayscale
./test_rotation
./test_blur
./test_parallel 32  # 32 изображения для теста
```

## Интеграция с Python

Библиотека интегрирована с Flask веб-интерфейсом через subprocess:

```python
import subprocess

result = subprocess.run([
    './image_processor',
    'grayscale',
    'input.png',
    'output.png'
], capture_output=True)
```

Подробнее: [web/README_WEB.md](web/README_WEB.md)

## Лицензия

Курсовой проект по параллельному программированию.

## Авторы

Команда из 4 студентов | CUDA + NVIDIA GPU

