#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA Image Processing Web Interface
Веб-интерфейс для обработки изображений на GPU
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import subprocess
import uuid
from werkzeug.utils import secure_filename
import time
import zipfile
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
CUDA_EXECUTABLE = '../build/image_processor'

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
    try:
        # Формируем команду
        cmd = [CUDA_EXECUTABLE, filter_name, input_path, output_path]
        
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
            elif filter_name in ['blur_gaussian', 'blur_separable'] and 'sigma' in params:
                cmd.append(str(params['sigma']))
            elif filter_name == 'blur_motion':
                if 'length' in params:
                    cmd.append(str(params['length']))
                if 'angle' in params:
                    cmd.append(str(params['angle']))
            elif filter_name == 'rotate_arbitrary' and 'angle' in params:
                cmd.append(str(params['angle']))
        
        # Запускаем и замеряем время
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, "Успешно обработано", execution_time
        else:
            return False, f"Ошибка: {result.stderr}", 0
            
    except subprocess.TimeoutExpired:
        return False, "Превышено время ожидания (30 сек)", 0
    except Exception as e:
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
            mode = request.form.get('rotation_mode', '90')
            
            # Определяем команду в зависимости от режима
            if mode == '90':
                filter_name = 'rotate90'
            elif mode == '180':
                filter_name = 'rotate180'
            elif mode == '270':
                filter_name = 'rotate270'
            elif mode == 'custom':
                # Произвольный угол
                filter_name = 'rotate_arbitrary'
                angle = int(request.form.get('rotation_angle', 45))
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
                    'name': 'rotation_mode',
                    'label': 'Режим поворота',
                    'type': 'buttons',
                    'options': [
                        {'value': '90', 'label': '90°', 'icon': '↻'},
                        {'value': '180', 'label': '180°', 'icon': '↻↻'},
                        {'value': '270', 'label': '270°', 'icon': '↺'},
                        {'value': 'custom', 'label': 'Произвольный', 'icon': '🎯'}
                    ],
                    'default': '90'
                },
                {
                    'name': 'rotation_angle',
                    'label': 'Произвольный угол',
                    'type': 'range',
                    'min': 0,
                    'max': 360,
                    'step': 1,
                    'default': 45,
                    'unit': '°',
                    'depends_on': 'rotation_mode',
                    'depends_value': 'custom'
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

if __name__ == '__main__':
    # Создаем папки если не существуют
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    print("=" * 60)
    print("🚀 CUDA Image Processing Web Interface")
    print("=" * 60)
    print("📍 Сервер запущен: http://localhost:5000")
    print("🖼️  Загружайте изображения и обрабатывайте на GPU!")
    print("💡 Для остановки нажмите Ctrl+C")
    print("=" * 60)
    print()
    
    # Запускаем в режиме разработки (для курсовой проекта это нормально)
    app.run(debug=False, host='127.0.0.1', port=5000)

