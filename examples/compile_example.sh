#!/bin/bash

# Скрипт для компиляции примера использования библиотеки

echo "=== Компиляция примера использования библиотеки ==="
echo ""

# Путь к корню проекта
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Путь к проекту: $PROJECT_ROOT"
echo ""

# Проверка наличия библиотеки
if [ ! -f "$PROJECT_ROOT/build/libimage_processing.a" ]; then
    echo "❌ Библиотека не найдена!"
    echo "Сначала собери библиотеку:"
    echo "  cd $PROJECT_ROOT/build"
    echo "  cmake .."
    echo "  make -j4"
    exit 1
fi

echo "✓ Библиотека найдена"
echo ""

# Компиляция
echo "🔨 Компиляция simple_example.cu..."
echo ""

nvcc simple_example.cu \
    -I"$PROJECT_ROOT/src" \
    -L"$PROJECT_ROOT/build" \
    -limage_processing \
    -lcudart \
    -allow-unsupported-compiler \
    -o simple_example

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Компиляция успешна!"
    echo ""
    echo "Запуск:"
    echo "  ./simple_example input.jpg output.jpg"
    echo ""
    echo "Пример:"
    echo "  ./simple_example ../foto/test.jpg result_gray.jpg"
else
    echo ""
    echo "❌ Ошибка компиляции"
    exit 1
fi

