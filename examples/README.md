# Примеры использования библиотеки

Эта папка содержит примеры использования библиотеки `libimage_processing.a`.

## Быстрый старт

### 1. Убедись что библиотека собрана

```bash
cd ..
mkdir -p build && cd build
cmake ..
make -j4
cd ../examples
```

### 2. Скомпилируй пример

```bash
./compile_example.sh
```

### 3. Запусти пример

```bash
# Базовое использование
./simple_example input.jpg output.jpg

# Пример с реальным изображением
./simple_example ../foto/WIN_20251013_13_01_54_Pro.png result_gray.png
```

## Что делает пример?

`simple_example.cu` показывает базовый workflow:

1. **Загрузка изображения** с диска
2. **Применение фильтра** grayscale на GPU
3. **Копирование результата** с GPU на CPU
4. **Сохранение** обработанного изображения

## Результат

После запуска:
```
=== Простой пример использования библиотеки ===

Загрузка изображения: ../foto/WIN_20251013_13_01_54_Pro.png
Изображение загружено: 1280x720, 3 каналов

Применение фильтра grayscale...
Фильтр применен

Копирование результата с GPU...
Данные скопированы

Сохранение результата: result_gray.png
Результат сохранен

Готово! Проверь файл: result_gray.png
```

## Создание своего примера

Используй `simple_example.cu` как шаблон:

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

Компиляция:

```bash
nvcc my_example.cu \
    -I../src \
    -L../build \
    -limage_processing \
    -lcudart \
    -allow-unsupported-compiler \
    -o my_example
```

## Другие примеры

Смотри также:
- `../tests/test_grayscale.cu` - тест оттенков серого
- `../tests/test_rotation.cu` - тест поворота
- `../tests/test_blur.cu` - тест размытия
- `../tests/test_parallel.cu` - тест параллельной обработки

## Документация

- [../INSTALL.md](../INSTALL.md) - Полная инструкция по установке
- [../LIBRARY.md](../LIBRARY.md) - API документация
- [../README.md](../README.md) - Общее описание проекта

