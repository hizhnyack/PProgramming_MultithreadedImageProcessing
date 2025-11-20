#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA Image Processing Web Interface
Веб-интерфейс для обработки изображений на GPU
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Определяем путь к CUDA программе относительно директории скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CUDA_EXECUTABLE = os.path.join(PROJECT_ROOT, 'build', 'image_processor')
WRAPPER_SCRIPT = os.path.join(PROJECT_ROOT, 'run_with_nvidia.sh')

def allowed_file(filename):
    """Проверка допустимого расширения файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_cuda_filter(input_path, output_path, filter_name, params=None):
    """
    Запускает CUDA программу для обработки изображения
    
    Args:
        input_path: путь к входному файлу
        output_path: путь к выходному файлу
        filter_name: название фильтра (grayscale, rotate90, blur, etc.)
        params: дополнительные параметры (например, радиус размытия)
    
    Returns:
        (success, message, execution_time)
    """
    import logging
    from datetime import datetime
    
    # Настройка логирования в файл
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_file = os.path.join(LOG_DIR, f'web_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Также выводим в консоль
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    try:
        logger.info(f"[WEB] run_cuda_filter called: filter={filter_name}, input={input_path}, output={output_path}")
        
        # Проверяем существование файлов
        if not os.path.exists(input_path):
            logger.error(f"[WEB] Input file not found: {input_path}")
            return False, f"Входной файл не найден: {input_path}", 0
        
        logger.info(f"[WEB] Input file exists: {input_path} ({os.path.getsize(input_path)} bytes)")
        
        # Формируем команду
        # Используем wrapper скрипт для автоматической активации NVIDIA GPU
        use_wrapper = os.path.exists(WRAPPER_SCRIPT) and os.access(WRAPPER_SCRIPT, os.X_OK)
        logger.info(f"[WEB] Wrapper script exists: {use_wrapper}, path: {WRAPPER_SCRIPT}")
        logger.info(f"[WEB] CUDA executable: {CUDA_EXECUTABLE}, exists: {os.path.exists(CUDA_EXECUTABLE)}")
        
        if use_wrapper:
            cmd = [WRAPPER_SCRIPT, CUDA_EXECUTABLE, filter_name, input_path, output_path]
            logger.info(f"[WEB] Using wrapper script: {WRAPPER_SCRIPT}")
        else:
            # Если wrapper нет, запускаем напрямую
            cmd = [CUDA_EXECUTABLE, filter_name, input_path, output_path]
            logger.info(f"[WEB] Running CUDA executable directly")
        
        # Добавляем параметры если есть
        if params:
            if filter_name == 'blur' and 'radius' in params:
                cmd.append(str(params['radius']))
                logger.info(f"[WEB] Blur radius: {params['radius']}")
        
        logger.info(f"[WEB] Command: {' '.join(cmd)}")
        
        # Устанавливаем переменные окружения для NVIDIA GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['__NV_PRIME_RENDER_OFFLOAD'] = '1'
        env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
        
        logger.info(f"[WEB] Environment variables:")
        logger.info(f"[WEB]   CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"[WEB]   __NV_PRIME_RENDER_OFFLOAD={env.get('__NV_PRIME_RENDER_OFFLOAD')}")
        logger.info(f"[WEB]   __GLX_VENDOR_LIBRARY_NAME={env.get('__GLX_VENDOR_LIBRARY_NAME')}")
        
        # Запускаем и замеряем время
        logger.info(f"[WEB] Starting subprocess...")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        execution_time = time.time() - start_time
        
        logger.info(f"[WEB] Process finished: returncode={result.returncode}, time={execution_time:.3f}s")
        
        # Логируем полный вывод в файл, но ограничиваем для возврата пользователю
        logger.debug(f"[WEB] Full stdout:\n{result.stdout}")
        logger.debug(f"[WEB] Full stderr:\n{result.stderr}")
        
        # Для сообщения пользователю берем последние строки stderr (где обычно ошибки)
        error_preview = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
        
        if result.returncode == 0:
            logger.info(f"[WEB] Success!")
            return True, "Успешно обработано", execution_time
        else:
            # Используем preview для сообщения пользователю
            error_msg = error_preview if error_preview else result.stdout
            logger.error(f"[WEB] Process failed. Full error logged to: {log_file}")
            logger.error(f"[WEB] Error preview: {error_msg[:500]}")
            return False, f"Ошибка: {error_msg[:500]}", 0
            
    except subprocess.TimeoutExpired:
        logger.error(f"[WEB] Process timeout after 30 seconds")
        return False, "Превышено время ожидания (30 сек)", 0
    except Exception as e:
        logger.error(f"[WEB] Exception: {str(e)}", exc_info=True)
        return False, f"Ошибка выполнения: {str(e)}", 0

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка и обработка файла"""
    
    # Проверяем наличие файла
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Недопустимый формат файла. Используйте PNG, JPG, BMP'}), 400
    
    # Получаем параметры фильтра
    filter_name = request.form.get('filter', 'grayscale')
    
    # Генерируем уникальные имена файлов
    unique_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[1].lower()
    input_filename = f"{unique_id}_input.{ext}"
    output_filename = f"{unique_id}_output.png"
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    try:
        # Сохраняем загруженный файл
        file.save(input_path)
        
        # Получаем размер файла
        file_size = os.path.getsize(input_path) / 1024  # KB
        
        # Параметры фильтра
        params = {}
        
        # Обработка поворота - преобразуем угол в команду
        if filter_name == 'rotate':
            angle = int(request.form.get('rotation_angle', 90))
            # Определяем команду в зависимости от угла
            if angle == 90:
                filter_name = 'rotate90'
            elif angle == 180:
                filter_name = 'rotate180'
            elif angle == 270:
                filter_name = 'rotate270'
            else:
                # Для других углов используем rotate90 (можно расширить)
                filter_name = 'rotate90'
            params['angle'] = angle
        elif filter_name == 'blur':
            params['radius'] = int(request.form.get('blur_radius', 5))
        
        # Запускаем обработку
        success, message, exec_time = run_cuda_filter(input_path, output_path, filter_name, params)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'output_file': output_filename,
                'execution_time': round(exec_time, 3),
                'file_size': round(file_size, 2)
            })
        else:
            return jsonify({'success': False, 'error': message}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка сервера: {str(e)}'}), 500
    finally:
        # Удаляем входной файл (экономим место)
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание обработанного файла"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'Файл не найден'}), 404

@app.route('/view/<filename>')
def view_file(filename):
    """Просмотр обработанного файла"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Файл не найден'}), 404

@app.route('/filters')
def get_filters():
    """Получить список доступных фильтров"""
    filters = [
        {
            'id': 'grayscale',
            'name': 'Оттенки серого',
            'description': 'Преобразование в черно-белое изображение',
            'icon': '🎨',
            'params': []
        },
        {
            'id': 'rotate',
            'name': 'Поворот',
            'description': 'Поворот изображения на любой угол',
            'icon': '🔄',
            'params': [
                {
                    'name': 'rotation_angle',
                    'label': 'Угол поворота',
                    'type': 'range',
                    'min': 0,
                    'max': 360,
                    'step': 90,
                    'default': 90,
                    'unit': '°'
                }
            ]
        },
        {
            'id': 'blur',
            'name': 'Размытие',
            'description': 'Применить эффект размытия',
            'icon': '🌫️',
            'params': [
                {
                    'name': 'blur_radius',
                    'label': 'Радиус размытия',
                    'type': 'range',
                    'min': 1,
                    'max': 20,
                    'default': 5,
                    'unit': 'px'
                }
            ]
        }
    ]
    return jsonify(filters)

def find_free_port(start_port=5000, max_attempts=10):
    """Находит свободный порт начиная с start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    # Создаем папки если не существуют
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Определяем порт (из переменной окружения или находим свободный)
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    # Проверяем, свободен ли порт
    free_port = find_free_port(port)
    if free_port != port:
        print(f"⚠ Порт {port} занят, используем порт {free_port}")
        port = free_port
    
    print("=" * 60)
    print("🚀 CUDA Image Processing Web Interface")
    print("=" * 60)
    print(f"📍 Сервер запущен: http://localhost:{port}")
    print("🖼️  Загружайте изображения и обрабатывайте на GPU!")
    print("💡 Для остановки нажмите Ctrl+C")
    print("=" * 60)
    print()
    
    # Запускаем в режиме разработки (для курсовой проекта это нормально)
    app.run(debug=False, host='127.0.0.1', port=port)

