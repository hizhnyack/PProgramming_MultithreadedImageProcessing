#!/bin/bash

# Скрипт проверки совместимости CUDA и драйвера NVIDIA

echo "=========================================="
echo "  Проверка совместимости CUDA/Драйвер"
echo "=========================================="
echo ""

# Проверка драйвера
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "📊 Текущая версия драйвера: $DRIVER_VERSION"
    
    # Извлекаем основную версию (например, 535 из 535.216.01)
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
    echo "   Основная версия: $DRIVER_MAJOR"
else
    echo "❌ nvidia-smi не найден"
    exit 1
fi

echo ""

# Проверка CUDA Toolkit
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
else
    echo "❌ CUDA Toolkit не найден"
    exit 1
fi

echo "📊 Версия CUDA Toolkit: $CUDA_VERSION"
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
echo "   Основная версия: $CUDA_MAJOR"

echo ""
echo "=========================================="
echo "  Требования совместимости:"
echo "=========================================="
echo ""

# Таблица совместимости
declare -A CUDA_DRIVER_REQ
CUDA_DRIVER_REQ["11.0"]=450
CUDA_DRIVER_REQ["11.1"]=455
CUDA_DRIVER_REQ["11.2"]=460
CUDA_DRIVER_REQ["11.3"]=465
CUDA_DRIVER_REQ["11.4"]=470
CUDA_DRIVER_REQ["11.5"]=470
CUDA_DRIVER_REQ["11.6"]=470
CUDA_DRIVER_REQ["11.7"]=470
CUDA_DRIVER_REQ["11.8"]=520
CUDA_DRIVER_REQ["12.0"]=525
CUDA_DRIVER_REQ["12.1"]=530
CUDA_DRIVER_REQ["12.2"]=535
CUDA_DRIVER_REQ["12.3"]=535
CUDA_DRIVER_REQ["12.4"]=550
CUDA_DRIVER_REQ["13.0"]=550

# Определяем минимальную требуемую версию драйвера
REQUIRED_DRIVER=""
if [ "$CUDA_MAJOR" -eq 11 ]; then
    REQUIRED_DRIVER=470
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    if [ -n "${CUDA_DRIVER_REQ[$CUDA_VERSION]}" ]; then
        REQUIRED_DRIVER=${CUDA_DRIVER_REQ[$CUDA_VERSION]}
    else
        REQUIRED_DRIVER=535
    fi
elif [ "$CUDA_MAJOR" -eq 13 ]; then
    REQUIRED_DRIVER=550
else
    REQUIRED_DRIVER=550
fi

echo "CUDA $CUDA_VERSION требует драйвер версии >= $REQUIRED_DRIVER"
echo ""

# Проверка совместимости
if [ "$DRIVER_MAJOR" -ge "$REQUIRED_DRIVER" ]; then
    echo "✅ Совместимость: ОК"
    echo "   Драйвер $DRIVER_MAJOR поддерживает CUDA $CUDA_VERSION"
else
    echo "❌ Совместимость: ОШИБКА"
    echo "   Драйвер $DRIVER_MAJOR НЕ поддерживает CUDA $CUDA_VERSION"
    echo "   Требуется драйвер версии >= $REQUIRED_DRIVER"
    echo ""
    echo "=========================================="
    echo "  Рекомендации:"
    echo "=========================================="
    echo ""
    
    if [ "$CUDA_MAJOR" -eq 13 ]; then
        echo "Для CUDA 13.0 рекомендуется установить драйвер 580:"
        echo "  sudo apt update"
        echo "  sudo apt install nvidia-driver-580"
        echo ""
        echo "Или драйвер 550 (минимальная версия):"
        echo "  sudo apt install nvidia-driver-550"
    else
        echo "Обновите драйвер до версии >= $REQUIRED_DRIVER"
    fi
    
    echo ""
    echo "После установки перезагрузите систему:"
    echo "  sudo reboot"
fi

echo ""

