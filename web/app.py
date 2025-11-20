#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA Image Processing Web Interface
Веб-интерфейс для обработки изображений на GPU
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import RequestEntityTooLarge
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np
import zipfile
from io import BytesIO

app = Flask(__name__)
# Увеличиваем лимит для пакетной загрузки: 3GB (для обработки многих изображений)
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024  # 3GB max request size
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

def get_dominant_color(image_path):
    """
    Вычисляет преобладающий цвет изображения для заполнения пустых углов при повороте
    
    Returns:
        int: значение цвета (0-255) для grayscale или среднее значение для RGB
    """
    try:
        img = Image.open(image_path)
        # Конвертируем в RGB если нужно
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Уменьшаем размер для ускорения вычислений
        img.thumbnail((100, 100))
        
        # Получаем данные изображения
        pixels = np.array(img)
        
        # Вычисляем средний цвет
        if len(pixels.shape) == 3:
            # RGB изображение - вычисляем среднее по всем каналам
            avg_color = np.mean(pixels, axis=(0, 1))
            # Возвращаем среднее значение для использования как grayscale
            dominant = int(np.mean(avg_color))
        else:
            # Grayscale
            dominant = int(np.mean(pixels))
        
        return dominant
    except Exception as e:
        # В случае ошибки возвращаем черный цвет
        return 0

@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(e):
    """Обработка ошибки превышения размера запроса"""
    max_size_gb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024 * 1024)
    return jsonify({
        'success': False,
        'error': f'Размер загружаемых файлов слишком большой! Максимальный размер запроса: {max_size_gb:.0f} ГБ. Попробуйте загрузить меньше файлов или уменьшите их размер.'
    }), 413

def run_cuda_filter(input_path, output_path, filter_name, params=None, processor_mode='gpu', timeout_seconds=None):
    """
    Запускает CUDA или CPU программу для обработки изображения
    
    Args:
        input_path: путь к входному файлу
        output_path: путь к выходному файлу
        filter_name: название фильтра (grayscale, rotate90, blur, etc.)
        params: дополнительные параметры (например, радиус размытия)
        processor_mode: 'gpu' или 'cpu'
    
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
        
        if use_wrapper and processor_mode == 'gpu':
            cmd = [WRAPPER_SCRIPT, CUDA_EXECUTABLE, filter_name, input_path, output_path]
            logger.info(f"[WEB] Using wrapper script: {WRAPPER_SCRIPT}")
        else:
            # Если wrapper нет или CPU режим, запускаем напрямую
            cmd = [CUDA_EXECUTABLE, filter_name, input_path, output_path]
            logger.info(f"[WEB] Running executable directly (mode: {processor_mode})")
        
        # Добавляем параметры если есть
        if params:
            if filter_name == 'grayscale_weighted':
                if 'r_weight' in params:
                    cmd.append(str(params['r_weight']))
                if 'g_weight' in params:
                    cmd.append(str(params['g_weight']))
                if 'b_weight' in params:
                    cmd.append(str(params['b_weight']))
            elif filter_name == 'blur' and 'radius' in params:
                cmd.append(str(params['radius']))
                logger.info(f"[WEB] Blur radius: {params['radius']}")
            elif filter_name in ['blur_gaussian', 'blur_separable'] and 'sigma' in params:
                cmd.append(str(params['sigma']))
            elif filter_name == 'blur_motion':
                if 'length' in params:
                    cmd.append(str(params['length']))
                if 'angle' in params:
                    cmd.append(str(params['angle']))
            elif filter_name == 'rotateArbitrary' and 'angle' in params:
                # Для произвольного поворота добавляем угол и вычисляем преобладающий цвет
                angle = params['angle']
                dominant_color = get_dominant_color(input_path)
                cmd.append(str(angle))
                cmd.append(str(dominant_color))
                logger.info(f"[WEB] Rotation angle: {angle}°, dominant color: {dominant_color}")
            elif filter_name == 'rotate_arbitrary' and 'angle' in params:
                cmd.append(str(params['angle']))
        
        # Добавляем флаг --cpu если используется CPU режим
        if processor_mode == 'cpu':
            cmd.append('--cpu')
            logger.info(f"[WEB] CPU mode enabled")
        
        logger.info(f"[WEB] Command: {' '.join(cmd)}")
        
        # Устанавливаем переменные окружения для NVIDIA GPU (только для GPU режима)
        env = os.environ.copy()
        if processor_mode == 'gpu':
            env['CUDA_VISIBLE_DEVICES'] = '0'
            env['__NV_PRIME_RENDER_OFFLOAD'] = '1'
            env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
        
        logger.info(f"[WEB] Environment variables:")
        logger.info(f"[WEB]   CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"[WEB]   __NV_PRIME_RENDER_OFFLOAD={env.get('__NV_PRIME_RENDER_OFFLOAD')}")
        logger.info(f"[WEB]   __GLX_VENDOR_LIBRARY_NAME={env.get('__GLX_VENDOR_LIBRARY_NAME')}")
        
        # Запускаем и замеряем время
        # Таймаут зависит от режима: для CPU и пакетной обработки нужен больший таймаут
        if timeout_seconds is None:
            timeout_seconds = 300 if processor_mode == 'cpu' else 30  # 5 минут для CPU, 30 сек для GPU
        logger.info(f"[WEB] Starting subprocess... (timeout: {timeout_seconds}s)")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, env=env)
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
        # Используем переданный таймаут или вычисляем по умолчанию
        actual_timeout = timeout_seconds if timeout_seconds is not None else (300 if processor_mode == 'cpu' else 30)
        logger.error(f"[WEB] Process timeout after {actual_timeout} seconds")
        return False, f"Превышено время ожидания ({actual_timeout} сек)", 0
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
    processor_mode = request.form.get('processor', 'gpu')  # 'gpu' или 'cpu'
    
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
        
        # Обработка grayscale
        if filter_name == 'grayscale':
            mode = request.form.get('grayscale_mode', 'standard')
            
            if mode == 'weighted':
                filter_name = 'grayscale_weighted'
                params['r_weight'] = float(request.form.get('weight_r', 0.299))
                params['g_weight'] = float(request.form.get('weight_g', 0.587))
                params['b_weight'] = float(request.form.get('weight_b', 0.114))
        
        # Обработка поворота
        if filter_name == 'rotate':
            angle = float(request.form.get('rotation_angle', 90))
            # Нормализуем угол в диапазон 0-360
            angle = angle % 360
            if angle < 0:
                angle += 360
            
            # Для стандартных углов используем оптимизированные функции
            if angle == 0 or angle == 360:
                filter_name = 'rotate0'  # Без поворота
            elif angle == 90:
                filter_name = 'rotate90'
            elif angle == 180:
                filter_name = 'rotate180'
            elif angle == 270:
                filter_name = 'rotate270'
            else:
                # Для произвольных углов используем rotateArbitrary
                filter_name = 'rotateArbitrary'
            params['angle'] = angle
        elif filter_name == 'blur':
            # Получаем выбранный алгоритм
            algorithm = request.form.get('blur_algorithm', 'box')
            radius = int(request.form.get('blur_radius', 5))
            
            # Определяем команду и параметры в зависимости от алгоритма
            if algorithm == 'box':
                filter_name = 'blur'
                params['radius'] = radius
            elif algorithm == 'gaussian':
                filter_name = 'blur_gaussian'
                params['sigma'] = radius / 2.0  # Преобразуем радиус в sigma
            elif algorithm == 'separable':
                filter_name = 'blur_separable'
                params['sigma'] = radius / 2.0
            elif algorithm == 'motion':
                filter_name = 'blur_motion'
                params['length'] = radius * 2  # Длина размытия
                params['angle'] = int(request.form.get('motion_angle', 0))
        
        # Запускаем обработку
        success, message, exec_time = run_cuda_filter(input_path, output_path, filter_name, params, processor_mode)
        
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

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    """Пакетная загрузка и обработка файлов"""
    
    # Проверяем наличие файлов
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'Файлы не найдены'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'success': False, 'error': 'Файлы не выбраны'}), 400
    
    # Фильтруем валидные файлы
    valid_files = []
    for file in files:
        if file.filename and allowed_file(file.filename):
            valid_files.append(file)
    
    if not valid_files:
        return jsonify({'success': False, 'error': 'Нет валидных файлов. Используйте PNG, JPG, BMP'}), 400
    
    # Получаем параметры фильтра
    filter_name = request.form.get('filter', 'grayscale')
    processor_mode = request.form.get('processor', 'gpu')  # 'gpu' или 'cpu'
    
    # Параметры фильтра
    params = {}
    if filter_name == 'rotate':
        angle = float(request.form.get('rotation_angle', 90))
        # Нормализуем угол в диапазон 0-360
        angle = angle % 360
        if angle < 0:
            angle += 360
        
        # Для стандартных углов используем оптимизированные функции
        if angle == 0 or angle == 360:
            filter_name = 'rotate0'  # Без поворота
        elif angle == 90:
            filter_name = 'rotate90'
        elif angle == 180:
            filter_name = 'rotate180'
        elif angle == 270:
            filter_name = 'rotate270'
        else:
            # Для произвольных углов используем rotateArbitrary
            filter_name = 'rotateArbitrary'
        params['angle'] = angle
    elif filter_name == 'blur':
        params['radius'] = int(request.form.get('blur_radius', 5))
    
    # Обрабатываем файлы параллельно
    results = []
    success_count = 0
    failed_count = 0
    
    def process_single_file(file):
        """Обработка одного файла"""
        try:
            # Генерируем уникальные имена файлов
            unique_id = str(uuid.uuid4())
            ext = file.filename.rsplit('.', 1)[1].lower()
            input_filename = f"{unique_id}_input.{ext}"
            output_filename = f"{unique_id}_output.png"
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Сохраняем файл
            file.save(input_path)
            
            # Запускаем обработку
            # Для пакетной обработки увеличиваем таймаут
            # Вычисляем динамический таймаут: ~3 секунды на файл для CPU, ~1 секунда для GPU
            num_files = len(valid_files)
            if processor_mode == 'cpu':
                timeout_per_file = 3  # секунд на файл для CPU
            else:
                timeout_per_file = 1  # секунд на файл для GPU
            dynamic_timeout = max(300, num_files * timeout_per_file)  # минимум 5 минут
            
            start_time = time.time()
            success, message, exec_time = run_cuda_filter(input_path, output_path, filter_name, params, processor_mode, dynamic_timeout)
            execution_time = time.time() - start_time
            
            # Удаляем входной файл
            if os.path.exists(input_path):
                os.remove(input_path)
            
            return {
                'filename': file.filename,
                'success': success,
                'output_file': output_filename if success else None,
                'execution_time': execution_time,
                'error': message if not success else None
            }
        except Exception as e:
            return {
                'filename': file.filename,
                'success': False,
                'output_file': None,
                'execution_time': 0,
                'error': str(e)
            }
    
    # Используем ThreadPoolExecutor для параллельной обработки
    # Количество потоков = количество CPU ядер
    max_workers = min(len(valid_files), os.cpu_count() or 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем обработку всех файлов
        future_to_file = {executor.submit(process_single_file, file): file for file in valid_files}
        
        # Собираем результаты по мере завершения
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            
            if result['success']:
                success_count += 1
            else:
                failed_count += 1
    
    return jsonify({
        'success': True,
        'results': results,
        'success_count': success_count,
        'failed_count': failed_count,
        'total_count': len(valid_files)
    })

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
            'description': 'Преобразование в черно-белое',
            'icon': '🎨',
            'params': [
                {
                    'name': 'grayscale_mode',
                    'label': 'Алгоритм преобразования',
                    'type': 'buttons',
                    'options': [
                        {'value': 'standard', 'label': 'Стандартный', 'icon': '⚡'},
                        {'value': 'weighted', 'label': 'Настраиваемый', 'icon': '⚙️'}
                    ],
                    'default': 'standard'
                },
                {
                    'name': 'weight_r',
                    'label': 'Вес красного (R)',
                    'type': 'range',
                    'min': 0,
                    'max': 1,
                    'step': 0.01,
                    'default': 0.299,
                    'unit': '',
                    'depends_on': 'grayscale_mode',
                    'depends_value': 'weighted'
                },
                {
                    'name': 'weight_g',
                    'label': 'Вес зеленого (G)',
                    'type': 'range',
                    'min': 0,
                    'max': 1,
                    'step': 0.01,
                    'default': 0.587,
                    'unit': '',
                    'depends_on': 'grayscale_mode',
                    'depends_value': 'weighted'
                },
                {
                    'name': 'weight_b',
                    'label': 'Вес синего (B)',
                    'type': 'range',
                    'min': 0,
                    'max': 1,
                    'step': 0.01,
                    'default': 0.114,
                    'unit': '',
                    'depends_on': 'grayscale_mode',
                    'depends_value': 'weighted'
                }
            ]
        },
        {
            'id': 'rotate',
            'name': 'Поворот',
            'description': 'Быстрый поворот или произвольный угол',
            'icon': '🔄',
            'params': [
                {
                    'name': 'rotation_angle',
                    'label': 'Угол поворота',
                    'type': 'range',
                    'min': 0,
                    'max': 360,
                    'step': 1,
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
                    'name': 'blur_algorithm',
                    'label': 'Алгоритм размытия',
                    'type': 'select',
                    'options': [
                        {'value': 'box', 'label': 'Box Blur (быстрый, простой)'},
                        {'value': 'gaussian', 'label': 'Gaussian Blur (качественный)'},
                        {'value': 'separable', 'label': 'Separable Gaussian (оптимизированный)'},
                        {'value': 'motion', 'label': 'Motion Blur (эффект движения)'}
                    ],
                    'default': 'box'
                },
                {
                    'name': 'blur_radius',
                    'label': 'Радиус/Интенсивность',
                    'type': 'range',
                    'min': 1,
                    'max': 20,
                    'default': 5,
                    'unit': 'px'
                },
                {
                    'name': 'motion_angle',
                    'label': 'Угол движения (для Motion Blur)',
                    'type': 'range',
                    'min': 0,
                    'max': 360,
                    'step': 45,
                    'default': 0,
                    'unit': '°',
                    'depends_on': 'blur_algorithm',
                    'depends_value': 'motion'
                }
            ]
        }
    ]
    return jsonify(filters)

@app.route('/download_all', methods=['POST'])
def download_all():
    """Скачать все обработанные файлы в ZIP архиве"""
    try:
        data = request.get_json()
        filenames = data.get('files', [])
        
        if not filenames:
            return jsonify({'success': False, 'error': 'Нет файлов для скачивания'}), 400
        
        # Создаем ZIP архив в памяти
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in filenames:
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if os.path.exists(file_path):
                    # Добавляем файл в архив с его именем
                    zf.write(file_path, arcname=filename)
        
        memory_file.seek(0)
        
        # Генерируем имя для ZIP файла
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        zip_filename = f'processed_images_{timestamp}.zip'
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
