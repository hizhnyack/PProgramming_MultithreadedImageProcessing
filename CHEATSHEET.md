# Шпаргалка по использованию библиотеки

## Быстрый старт

### Сборка библиотеки
```bash
mkdir build && cd build
cmake ..
make -j4
```

Результат: `build/libimage_processing.a` (1.2 MB)

### Использование готовой программы
```bash
cd build

# Grayscale
./image_processor grayscale input.png output.png

# Поворот
./image_processor rotate90 input.png output.png

# Размытие
./image_processor blur input.png output.png 5

# Пакетная обработка
./image_processor batch grayscale ../foto ../output
```

## Компиляция своей программы

### Простая компиляция
```bash
nvcc my_program.cu \
    -I/path/to/src \
    -L/path/to/build \
    -limage_processing \
    -lcudart \
    -allow-unsupported-compiler \
    -o my_program
```

### С CMake
```cmake
include_directories(/path/to/src)
link_directories(/path/to/build)
add_executable(my_program my_program.cu)
target_link_libraries(my_program PRIVATE image_processing cudart stdc++ m)
```

## Код: Минимальный пример

```cpp
#include "filters/grayscale.h"
#include "utils/image_loader.h"

int main() {
    ImageData image;
    ImageLoader::load("input.png", image);
    GrayscaleFilter::applyInPlace(image);
    cudaMemcpy(image.data, image.gpu_data, 
               image.size_bytes, cudaMemcpyDeviceToHost);
    ImageLoader::save("output.png", image);
    return 0;
}
```

## Доступные фильтры

### Grayscale
```cpp
GrayscaleFilter::applyInPlace(image);
GrayscaleFilter::apply(input, output);
GrayscaleFilter::applyWithMethod(image, GrayscaleMethod::LUMINOSITY);
```

### Rotation
```cpp
RotationFilter::rotate90(input, output);
RotationFilter::rotate180(input, output);
RotationFilter::rotate270(input, output);
RotationFilter::rotateArbitrary(input, output, 45.0f);
```

### Blur
```cpp
BlurFilter::applyBoxBlur(image, 5);
BlurFilter::applyGaussianBlur(image, 5, 2.0f);
BlurFilter::applyMotionBlur(image, 10, 45.0f);
```

## Параллельная обработка

```cpp
#include "core/parallel_processor.h"

// Инициализация
ParallelProcessor::ParallelConfig config;
config.maxConcurrentStreams = 4;
ParallelProcessor::initialize(config);

// Обработка
std::vector<std::vector<unsigned char>> inputImages, outputImages;
auto stats = ParallelProcessor::processBatchParallel(
    inputImages, outputImages, width, height, "grayscale"
);

stats.print();  // Вывод статистики
ParallelProcessor::cleanup();
```

## Тестирование

```bash
cd build

./test_grayscale      # Тест оттенков серого
./test_rotation       # Тест поворота
./test_blur           # Тест размытия
./test_parallel 32    # Тест параллельной обработки (32 изображения)
```

## Веб-интерфейс

```bash
cd web
./start_server.sh
# Открой: http://127.0.0.1:5000
```

## Производительность

| Изображений | Последовательно | Параллельно | Ускорение |
|-------------|-----------------|-------------|-----------|
| 16          | 494 ms          | 399 ms      | 1.24x     |
| 32          | 987 ms          | 608 ms      | 1.62x     |
| 64          | 1936 ms         | 927 ms      | 2.09x     |

## Структура проекта

```
PProgramming_MultithreadedImageProcessing/
├── build/
│   └── libimage_processing.a  ← Библиотека
├── src/
│   ├── core/                   ← Ядро
│   ├── filters/                ← Фильтры
│   └── utils/                  ← Утилиты
├── examples/                   ← Примеры
├── tests/                      ← Тесты
└── web/                        ← Веб-интерфейс
```

## Документация

- **INSTALL.md** - Полная инструкция по установке
- **LIBRARY.md** - API документация
- **README.md** - Описание проекта
- **examples/README.md** - Примеры использования

## Частые проблемы

### Библиотека не найдена
```bash
# Проверь что она собрана
ls -lh build/libimage_processing.a
```

### CUDA ошибка
```bash
# Проверь GPU
nvidia-smi

# Проверь CUDA
nvcc --version
```

### Ошибка компиляции
```bash
# Добавь флаг
-allow-unsupported-compiler
```

## Полезные команды

```bash
# Информация о GPU
nvidia-smi

# Версия CUDA
nvcc --version

# Содержимое библиотеки
ar -t build/libimage_processing.a

# Размер библиотеки
du -h build/libimage_processing.a

# Очистка сборки
rm -rf build && mkdir build
```

---

**Готово!** Теперь у тебя есть всё для работы с библиотекой! 

