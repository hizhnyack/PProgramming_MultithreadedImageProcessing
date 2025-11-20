# CUDA Image Processing Library

**Курсовой проект по дисциплине "Параллельное программирование"**  
Команда из 4 студентов | Срок реализации: 2 недели

Этот проект представляет собой минималистичную библиотеку для ускоренной обработки изображений с использованием **CUDA**. Цель — продемонстрировать преимущества параллельных вычислений на GPU при выполнении типовых операций над изображениями.

---

## Возможности

На данный момент реализованы следующие операции:
- **Преобразование в оттенки серого** (`grayscale`) - 3 варианта (стандартный, взвешенный, in-place)
- **Поворот изображения** (`rotation`) - 90°, 180°, 270° и произвольный угол
- **Размытие** (`blur`) - Box, Gaussian, Separable Gaussian, Motion blur
- **Пакетная обработка** - многопоточная обработка директорий с изображениями
### Производительность и оптимизация
- **Параллельная обработка**: Добавлен `ParallelProcessor` класс для одновременной обработки нескольких изображений с использованием CUDA streams
- **Оптимизация памяти**: Улучшено управление GPU памятью с использованием pinned memory для быстрых transfers
- **Статистика выполнения**: Встроенный мониторинг времени выполнения и использования потоков

Все операции выполняются на GPU с использованием собственных CUDA-ядер. Библиотека поддерживает форматы PNG, JPG, BMP через STB библиотеку.

---

## Архитектура проекта

```
PProgramming_MultithreadedImageProcessing/
├── src/
│   ├── core/                   # Ядро библиотеки
│   │   ├── image_processor.cu  # Управление GPU памятью, CUDA операции
│   │   ├── image_processor.h   # Публичный интерфейс (ImageData, ImageProcessor)
│   │   ├── cuda_kernels.cu     # Низкоуровневые CUDA ядра
│   │   └── cuda_utils.h        # Макросы для обработки ошибок CUDA
│   ├── filters/                # Фильтры обработки изображений
│   │   ├── grayscale.cu        # Преобразование в оттенки серого
│   │   ├── grayscale.h         # Интерфейс grayscale фильтра
│   │   ├── blur.cu             # Размытие (Box, Gaussian, Separable, Motion)
│   │   ├── blur.h              # Интерфейс blur фильтра
│   │   ├── rotation.cu         # Поворот изображения (90°, 180°, 270°, произвольный)
│   │   └── rotation.h          # Интерфейс rotation фильтра
│   ├── utils/                  # Утилиты
│   │   ├── image_loader.cu     # Загрузка/сохранение (PNG, JPG, BMP)
│   │   ├── image_loader.h      # Интерфейс загрузчика
│   │   ├── batch_processor.cu  # Многопоточная пакетная обработка
│   │   ├── batch_processor.h   # Интерфейс пакетного процессора
│   │   ├── stb_image.h         # Внешняя библиотека (загрузка изображений)
│   │   └── stb_image_write.h   # Внешняя библиотека (сохранение изображений)
│   └── main.cu                 # Главная программа (CLI)
├── tests/                      # Юнит-тесты
│   ├── test_grayscale.cu       # Тесты grayscale фильтра
│   ├── test_rotation.cu        # Тесты rotation фильтра
│   └── test_blur.cu            # Тесты blur фильтра
├── include/
│   └── cuda_image_lib.h        # Главный заголовочный файл библиотеки
├── samples/                    # Примеры использования
│   ├── simple_example.cu       # Простой пример обработки
│   ├── batch_example.cu        # Пакетная обработка
│   └── multi_filter_example.cu # Применение нескольких фильтров
├── CMakeLists.txt              # Конфигурация сборки
├── run_tests.sh                # Скрипт запуска тестов
└── .gitignore                  # Игнорируемые файлы
```

---

## Требования

- **ОС**: Linux (Ubuntu 22.04+) или WSL2
- **GPU**: NVIDIA с поддержкой CUDA (архитектура compute capability ≥ 8.6, например RTX 4060)
- **CUDA Toolkit**: версия 12.3 или выше
- **Компилятор**: GCC 12 (для совместимости с CUDA 12.3)
- **Сборка**: CMake 3.18+
- **Библиотеки**: STB (скачиваются автоматически)

---

## Сборка и запуск

### 1. Скачивание STB библиотек

```bash
cd src/utils
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
cd ../..
```

### 2. Сборка проекта

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
         -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
         -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
cmake --build .
```

### 3. Запуск тестов

```bash
./run_tests.sh
```

### 4. Использование

**Обработка одного изображения:**
```bash
./build/image_processor grayscale input.png output.png
./build/image_processor rotate90 input.png output.png
./build/image_processor blur input.png output.png 10
```

**Пакетная обработка директории:**
```bash
./build/image_processor batch grayscale ./images/ ./output/
```
