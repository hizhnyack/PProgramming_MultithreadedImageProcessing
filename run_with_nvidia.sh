#!/bin/bash
# Wrapper скрипт для запуска CUDA приложений с автоматической активацией NVIDIA GPU
# Использование: ./run_with_nvidia.sh <команда> [аргументы]

# Определяем директорию для логов
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/cuda_wrapper_$(date +%Y%m%d_%H%M%S).log"

# Создаем директорию для логов
mkdir -p "$LOG_DIR"

# Логирование в stderr и в файл
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [WRAPPER] $@"
    echo "$msg" >&2
    echo "$msg" >> "$LOG_FILE"
}

log "Wrapper script started"
log "Arguments: $@"

# Функция для активации NVIDIA GPU
activate_nvidia_gpu() {
    log "Checking NVIDIA GPU availability..."
    
    # Проверяем, доступен ли NVIDIA GPU через nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        log "nvidia-smi found, checking GPU..."
        if nvidia-smi &>/dev/null; then
            log "GPU is available via nvidia-smi"
            nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >&2
            return 0
        else
            log "WARNING: nvidia-smi failed"
        fi
    else
        log "WARNING: nvidia-smi not found in PATH"
    fi
    
    # Пытаемся загрузить модули NVIDIA если они не загружены
    if ! lsmod | grep -q "^nvidia "; then
        log "NVIDIA modules not loaded, attempting to load..."
        sudo modprobe nvidia 2>&1 | while read line; do log "modprobe nvidia: $line"; done
        sudo modprobe nvidia_uvm 2>&1 | while read line; do log "modprobe nvidia_uvm: $line"; done
        sleep 1
        log "Modules loaded, checking again..."
    else
        log "NVIDIA modules are loaded"
        lsmod | grep nvidia | head -5 | while read line; do log "  $line"; done
    fi
    
    # Проверяем устройства
    if [ -e /dev/nvidia0 ]; then
        log "NVIDIA device /dev/nvidia0 exists"
    else
        log "WARNING: /dev/nvidia0 does not exist"
    fi
    
    return 0
}

# Активируем GPU
activate_nvidia_gpu

# Устанавливаем переменные окружения для явного использования NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

log "Environment variables set:"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "  __NV_PRIME_RENDER_OFFLOAD=$__NV_PRIME_RENDER_OFFLOAD"
log "  __GLX_VENDOR_LIBRARY_NAME=$__GLX_VENDOR_LIBRARY_NAME"

# Проверяем CUDA перед запуском
if command -v nvcc &>/dev/null; then
    log "nvcc found: $(which nvcc)"
    nvcc --version | head -1 >&2
else
    log "WARNING: nvcc not found in PATH"
fi

log "Executing command: $@"
log "Log file: $LOG_FILE"
log "=========================================="

# В режиме on-demand используем prime-run для активации NVIDIA GPU
# prime-run автоматически активирует NVIDIA GPU для CUDA приложений
PRIME_RUN=""
if command -v prime-run &>/dev/null; then
    PRIME_RUN="prime-run"
elif [ -f "/usr/bin/prime-run" ]; then
    PRIME_RUN="/usr/bin/prime-run"
elif [ -f "/usr/local/bin/prime-run" ]; then
    PRIME_RUN="/usr/local/bin/prime-run"
elif [ -f "$SCRIPT_DIR/prime-run.sh" ]; then
    PRIME_RUN="$SCRIPT_DIR/prime-run.sh"
    log "Using local prime-run.sh wrapper"
fi

if [ -n "$PRIME_RUN" ]; then
    log "Using $PRIME_RUN to activate NVIDIA GPU for CUDA"
    log "Full command: $PRIME_RUN $@"
    log "=========================================="
    # Запускаем через prime-run с логированием вывода
    # prime-run устанавливает правильные переменные окружения для CUDA
    $PRIME_RUN "$@" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    log "Command finished with exit code: $exit_code"
    exit $exit_code
else
    log "WARNING: prime-run not found!"
    log "Attempting to use __NV_PRIME_RENDER_OFFLOAD method"
    log "=========================================="
    # Альтернативный метод: используем переменные окружения напрямую
    # Это может работать в некоторых системах
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export CUDA_VISIBLE_DEVICES=0
    log "Environment variables set for NVIDIA GPU"
    log "Executing: $@"
    log "=========================================="
    # Запускаем напрямую с логированием вывода
    "$@" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
    log "Command finished with exit code: $exit_code"
    exit $exit_code
fi

