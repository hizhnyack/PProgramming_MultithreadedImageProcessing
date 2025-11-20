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
            if filter_name == 'blur' and 'radius' in params:
                cmd.append(str(params['radius']))
        
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

