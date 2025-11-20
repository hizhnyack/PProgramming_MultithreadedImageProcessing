#!/bin/bash

# Скрипт быстрого запуска проекта CUDA Image Processing
# Автоматически проверяет зависимости, скачивает библиотеки, собирает и запускает проект

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка наличия NVIDIA GPU
check_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        if [ -n "$gpu_info" ]; then
            info "  ✓ NVIDIA GPU обнаружен: $gpu_info"
            return 0
        fi
    fi
    return 1
}

# Проверка зависимостей
check_dependencies() {
    info "Проверка зависимостей..."
    echo ""
    
    local missing_deps=()
    local cuda_missing=false
    local gpu_available=false
    
    # Проверка NVIDIA GPU
    if check_nvidia_gpu; then
        gpu_available=true
    else
        warning "  ⚠ NVIDIA GPU не обнаружен (nvidia-smi недоступен)"
        warning "     Проект требует NVIDIA GPU с поддержкой CUDA"
    fi
    
    # Проверка CUDA
    if ! command -v nvcc &> /dev/null; then
        # Проверяем, может быть CUDA установлена, но не в PATH
        if [ -f "/usr/local/cuda/bin/nvcc" ]; then
            warning "  ⚠ CUDA установлена, но не в PATH"
            warning "     Добавьте в ~/.bashrc или ~/.zshrc:"
            warning "     export PATH=/usr/local/cuda/bin:\$PATH"
            warning "     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
            warning "     Затем выполните: source ~/.bashrc"
            echo ""
            info "  Попытка использовать /usr/local/cuda/bin/nvcc..."
            if /usr/local/cuda/bin/nvcc --version &> /dev/null; then
                CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
                info "  ✓ CUDA Toolkit найден: версия $CUDA_VERSION (не в PATH)"
                warning "  ⚠ Рекомендуется добавить CUDA в PATH для удобства"
                # Временно добавляем в PATH для текущей сессии
                export PATH="/usr/local/cuda/bin:$PATH"
                export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
            else
                missing_deps+=("CUDA Toolkit (nvcc)")
                cuda_missing=true
            fi
        else
            missing_deps+=("CUDA Toolkit (nvcc)")
            cuda_missing=true
        fi
    else
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        info "  ✓ CUDA Toolkit найден: версия $CUDA_VERSION"
        
        # Проверка версии CUDA
        local cuda_major=$(echo "$CUDA_VERSION" | cut -d. -f1)
        local cuda_minor=$(echo "$CUDA_VERSION" | cut -d. -f2)
        
        if [ "$cuda_major" -lt 12 ] || ([ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -lt 0 ]); then
            warning "  ⚠ Рекомендуется CUDA 12.0+, найдена версия $CUDA_VERSION"
        else
            info "  ✓ Версия CUDA подходит для проекта"
        fi
    fi
    
    # Проверка CMake
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("CMake")
    else
        CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
        info "  ✓ CMake найден: версия $CMAKE_VERSION"
    fi
    
    # Проверка компиляторов
    if ! command -v gcc-12 &> /dev/null && ! command -v gcc &> /dev/null; then
        missing_deps+=("GCC (рекомендуется версия 12)")
    else
        if command -v gcc-12 &> /dev/null; then
            info "  ✓ GCC-12 найден"
        else
            GCC_VERSION=$(gcc --version | head -n1 | cut -d' ' -f4)
            info "  ✓ GCC найден: версия $GCC_VERSION"
        fi
    fi
    
    if ! command -v g++-12 &> /dev/null && ! command -v g++ &> /dev/null; then
        missing_deps+=("G++ (рекомендуется версия 12)")
    else
        if command -v g++-12 &> /dev/null; then
            info "  ✓ G++-12 найден"
        else
            info "  ✓ G++ найден"
        fi
    fi
    
    # Проверка wget (для скачивания STB)
    if ! command -v wget &> /dev/null; then
        missing_deps+=("wget")
    else
        info "  ✓ wget найден"
    fi
    
    echo ""
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        error "Отсутствуют следующие зависимости:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        
        # Специальные инструкции для CUDA
        if [ "$cuda_missing" = true ]; then
            echo "═══════════════════════════════════════════════════════════"
            echo "  Инструкции по установке CUDA Toolkit:"
            echo "═══════════════════════════════════════════════════════════"
            echo ""
            echo "1. Проверьте наличие NVIDIA GPU:"
            echo "   nvidia-smi"
            echo ""
            echo "2. Установите CUDA Toolkit 12.0 или выше:"
            echo ""
            echo "   СПОСОБ 1: Локальный deb-пакет (рекомендуется для Debian/Ubuntu):"
            echo ""
            echo "   а) Скачайте пакет с официального сайта:"
            echo "      https://developer.nvidia.com/cuda-downloads"
            echo ""
            echo "   б) Выберите вашу ОС и скачайте локальный deb-пакет"
            echo "      Например, для Debian 12:"
            echo "      wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-debian12-13-0-local_13.0.2-580.95.05-1_amd64.deb"
            echo ""
            echo "   в) Установите пакет:"
            echo "      sudo dpkg -i cuda-repo-debian12-13-0-local_13.0.2-580.95.05-1_amd64.deb"
            echo "      sudo cp /var/cuda-repo-debian12-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/"
            echo "      sudo apt-get update"
            echo "      sudo apt-get -y install cuda-toolkit-13-0"
            echo ""
            echo "   ⚠ ВАЖНО: После установки обязательно добавьте CUDA в PATH (см. шаг 3)"
            echo ""
            echo "   СПОСОБ 2: Через репозиторий (для Ubuntu 22.04):"
            echo ""
            echo "      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
            echo "      sudo dpkg -i cuda-keyring_1.1-1_all.deb"
            echo "      sudo apt-get update"
            echo "      sudo apt-get -y install cuda-toolkit-12-3"
            echo ""
            echo "3. Добавьте CUDA в PATH (добавьте в ~/.bashrc или ~/.zshrc):"
            echo "   export PATH=/usr/local/cuda/bin:\$PATH"
            echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
            echo ""
            echo "4. Перезагрузите терминал или выполните:"
            echo "   source ~/.bashrc  # или source ~/.zshrc"
            echo ""
            echo "5. Проверьте установку:"
            echo "   nvcc --version"
            echo "   nvidia-smi"
            echo ""
            
            if [ "$gpu_available" = false ]; then
                warning "⚠ ВНИМАНИЕ: NVIDIA GPU не обнаружен!"
                warning "   Этот проект требует NVIDIA GPU с поддержкой CUDA."
                warning "   Убедитесь, что:"
                warning "   - У вас установлена NVIDIA видеокарта"
                warning "   - Установлены драйверы NVIDIA"
                warning "   - GPU поддерживает CUDA (compute capability ≥ 8.6)"
                echo ""
            fi
        fi
        
        # Инструкции для других зависимостей
        if [[ " ${missing_deps[@]} " =~ " CMake " ]]; then
            echo "Установка CMake:"
            echo "  sudo apt-get install cmake"
            echo ""
        fi
        
        if [[ " ${missing_deps[@]} " =~ " wget " ]]; then
            echo "Установка wget:"
            echo "  sudo apt-get install wget"
            echo ""
        fi
        
        if [[ " ${missing_deps[@]} " =~ " GCC " ]] || [[ " ${missing_deps[@]} " =~ " G++ " ]]; then
            echo "Установка GCC/G++ 12:"
            echo "  sudo apt-get install gcc-12 g++-12"
            echo ""
        fi
        
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        error "Установите недостающие зависимости перед продолжением."
        exit 1
    fi
    
    success "Все зависимости установлены!"
    echo ""
}

# Скачивание STB библиотек
download_stb() {
    info "Проверка STB библиотек..."
    
    local stb_dir="src/utils"
    local stb_image="$stb_dir/stb_image.h"
    local stb_image_write="$stb_dir/stb_image_write.h"
    
    if [ ! -f "$stb_image" ] || [ ! -f "$stb_image_write" ]; then
        warning "STB библиотеки не найдены. Скачивание..."
        
        if [ ! -d "$stb_dir" ]; then
            mkdir -p "$stb_dir"
        fi
        
        cd "$stb_dir"
        
        if [ ! -f "stb_image.h" ]; then
            info "  Скачивание stb_image.h..."
            wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
        fi
        
        if [ ! -f "stb_image_write.h" ]; then
            info "  Скачивание stb_image_write.h..."
            wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
        fi
        
        cd - > /dev/null
        success "STB библиотеки загружены!"
    else
        info "  ✓ STB библиотеки уже присутствуют"
    fi
    echo ""
}

# Определение компиляторов
detect_compilers() {
    if command -v gcc-12 &> /dev/null && command -v g++-12 &> /dev/null; then
        CMAKE_C_COMPILER="/usr/bin/gcc-12"
        CMAKE_CXX_COMPILER="/usr/bin/g++-12"
        CMAKE_CUDA_HOST_COMPILER="/usr/bin/g++-12"
    elif command -v gcc &> /dev/null && command -v g++ &> /dev/null; then
        CMAKE_C_COMPILER=$(which gcc)
        CMAKE_CXX_COMPILER=$(which g++)
        CMAKE_CUDA_HOST_COMPILER=$(which g++)
        warning "Используются стандартные компиляторы (не gcc-12/g++-12). Могут возникнуть проблемы совместимости."
    else
        error "Не удалось найти компиляторы C/C++"
        exit 1
    fi
}

# Сборка проекта
build_project() {
    info "Сборка проекта..."
    
    detect_compilers
    
    # Создание директории build
    if [ ! -d "build" ]; then
        mkdir -p build
        info "  Создана директория build/"
    fi
    
    cd build
    
    # Конфигурация CMake
    info "  Конфигурация CMake..."
    cmake .. \
        -DCMAKE_C_COMPILER="$CMAKE_C_COMPILER" \
        -DCMAKE_CXX_COMPILER="$CMAKE_CXX_COMPILER" \
        -DCMAKE_CUDA_HOST_COMPILER="$CMAKE_CUDA_HOST_COMPILER" \
        2>&1 | grep -v "^--" || true
    
    # Сборка
    info "  Компиляция проекта..."
    if ! cmake --build . -j$(nproc) 2>&1 | tee build.log; then
        error "Ошибки при сборке. Проверьте build.log для деталей."
        echo ""
        echo "Последние строки лога:"
        tail -30 build.log | grep -E "(error|Error|ERROR|failed|Failed|FAILED)" || tail -20 build.log
        exit 1
    fi
    
    cd ..
    success "Проект успешно собран!"
    echo ""
}

# Запуск веб-сервера
start_web_server() {
    # Определяем корневую директорию проекта
    local project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local web_dir="$project_root/web"
    local start_script="$web_dir/start_server.sh"
    
    if [ ! -d "$web_dir" ]; then
        warning "Директория веб-интерфейса не найдена: $web_dir"
        return 1
    fi
    
    if [ ! -f "$start_script" ]; then
        warning "Скрипт запуска веб-сервера не найден: $start_script"
        return 1
    fi
    
    if [ ! -x "$start_script" ]; then
        info "Делаем скрипт исполняемым..."
        chmod +x "$start_script"
    fi
    
    echo ""
    info "Запуск веб-сервера..."
    echo ""
    
    # Запускаем веб-сервер (блокирующий вызов)
    # Скрипт start_server.sh сам обработает занятый порт
    cd "$web_dir"
    ./start_server.sh
}

# Главная функция
main() {
    echo "=========================================="
    echo "  CUDA Image Processing - Quick Start"
    echo "=========================================="
    echo ""
    
    # Проверка зависимостей
    check_dependencies
    
    # Скачивание STB
    download_stb
    
    # Сборка проекта
    build_project
    
    # Информация о запуске
    echo "=========================================="
    success "Проект готов к использованию!"
    echo "=========================================="
    echo ""
    
    # Проверяем, есть ли аргумент --no-web для пропуска веб-сервера
    local skip_web=false
    for arg in "$@"; do
        if [ "$arg" = "--no-web" ] || [ "$arg" = "-n" ]; then
            skip_web=true
            break
        fi
    done
    
    if [ "$skip_web" = false ]; then
        # Спрашиваем, запускать ли веб-сервер
        echo "Запустить веб-интерфейс? (Y/n)"
        read -t 5 -r response || response=""
        
        if [ -z "$response" ] || [ "$response" = "y" ] || [ "$response" = "Y" ] || [ "$response" = "yes" ] || [ "$response" = "Yes" ]; then
            echo ""
            start_web_server
        else
            echo ""
            echo "Доступные команды:"
            echo ""
            echo "1. Запуск тестов:"
            echo "   ./run_tests.sh"
            echo ""
            echo "2. Обработка одного изображения:"
            echo "   ./build/image_processor grayscale input.png output.png"
            echo "   ./build/image_processor blur input.png output.png 10"
            echo "   ./build/image_processor rotate90 input.png output.png"
            echo ""
            echo "3. Пакетная обработка:"
            echo "   ./build/image_processor batch grayscale ./images/ ./output/"
            echo "   ./build/image_processor batch-pipeline grayscale ./images/ ./output/"
            echo ""
            echo "4. Веб-интерфейс:"
            echo "   cd web && ./start_server.sh"
            echo ""
        fi
    else
        echo "Веб-сервер пропущен (использован флаг --no-web)"
        echo ""
        echo "Доступные команды:"
        echo ""
        echo "1. Запуск тестов:"
        echo "   ./run_tests.sh"
        echo ""
        echo "2. Обработка одного изображения:"
        echo "   ./build/image_processor grayscale input.png output.png"
        echo "   ./build/image_processor blur input.png output.png 10"
        echo "   ./build/image_processor rotate90 input.png output.png"
        echo ""
        echo "3. Пакетная обработка:"
        echo "   ./build/image_processor batch grayscale ./images/ ./output/"
        echo "   ./build/image_processor batch-pipeline grayscale ./images/ ./output/"
        echo ""
        echo "4. Веб-интерфейс:"
        echo "   cd web && ./start_server.sh"
        echo ""
    fi
}

# Запуск
main "$@"

