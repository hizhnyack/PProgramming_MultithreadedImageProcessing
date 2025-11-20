#!/bin/bash
# Простой wrapper для prime-run, если он не установлен
# Использует переменные окружения для активации NVIDIA GPU

# Устанавливаем переменные окружения для NVIDIA GPU
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export CUDA_VISIBLE_DEVICES=0

# Запускаем команду
exec "$@"

