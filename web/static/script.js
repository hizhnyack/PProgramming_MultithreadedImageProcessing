// Глобальные переменные
let selectedFile = null;
let selectedFilter = null;
let filters = [];
let outputFilename = null;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    loadFilters();
    setupDragAndDrop();
    setupFileInput();
});

// Загрузка списка фильтров
async function loadFilters() {
    try {
        const response = await fetch('/filters');
        filters = await response.json();
        renderFilters();
    } catch (error) {
        console.error('Ошибка загрузки фильтров:', error);
    }
}

// Отрисовка фильтров
function renderFilters() {
    const grid = document.getElementById('filtersGrid');
    grid.innerHTML = '';
    
    filters.forEach(filter => {
        const card = document.createElement('div');
        card.className = 'filter-card';
        card.onclick = () => selectFilter(filter);
        card.innerHTML = `
            <div class="filter-icon">${filter.icon}</div>
            <div class="filter-name">${filter.name}</div>
            <div class="filter-description">${filter.description}</div>
        `;
        card.dataset.filterId = filter.id;
        grid.appendChild(card);
    });
}

// Выбор фильтра
function selectFilter(filter) {
    selectedFilter = filter;
    
    // Обновляем визуальное выделение
    document.querySelectorAll('.filter-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`[data-filter-id="${filter.id}"]`).classList.add('selected');
    
    // Показываем параметры если есть
    const paramsDiv = document.getElementById('filterParams');
    if (filter.params && filter.params.length > 0) {
        paramsDiv.style.display = 'block';
        paramsDiv.innerHTML = '<h3>Параметры фильтра</h3>';
        
        filter.params.forEach(param => {
            const group = document.createElement('div');
            group.className = 'param-group';
            const unit = param.unit || '';
            
            // Создаем HTML для параметра
            let html = `
                <label>
                    ${param.label}
                    <span class="param-value" id="${param.name}_value">${param.default}${unit}</span>
                </label>
                <input 
                    type="${param.type}" 
                    id="${param.name}" 
                    name="${param.name}"
                    min="${param.min}" 
                    max="${param.max}" 
                    step="${param.step || 1}"
                    value="${param.default}"
                    oninput="document.getElementById('${param.name}_value').textContent = this.value + '${unit}'"
                >
            `;
            
            // Добавляем метки для ползунка поворота
            if (param.name === 'rotation_angle') {
                html += `
                    <div class="range-labels">
                        <span>0° (без поворота)</span>
                        <span>90°</span>
                        <span>180°</span>
                        <span>270°</span>
                        <span>360°</span>
                    </div>
                `;
            }
            
            group.innerHTML = html;
            paramsDiv.appendChild(group);
        });
    } else {
        paramsDiv.style.display = 'none';
    }
}

// Настройка Drag & Drop
function setupDragAndDrop() {
    const uploadBox = document.getElementById('uploadBox');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => {
            uploadBox.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => {
            uploadBox.classList.remove('dragover');
        }, false);
    });
    
    uploadBox.addEventListener('drop', handleDrop, false);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Настройка выбора файла
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });
}

// Обработка выбранного файла
function handleFile(file) {
    // Проверка типа файла
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        alert('Недопустимый формат файла! Используйте PNG, JPG или BMP.');
        return;
    }
    
    // Проверка размера (16 МБ)
    if (file.size > 16 * 1024 * 1024) {
        alert('Файл слишком большой! Максимальный размер: 16 МБ.');
        return;
    }
    
    selectedFile = file;
    
    // Показываем предпросмотр
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('previewImage').src = e.target.result;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('filterSection').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Обработка изображения
async function processImage() {
    if (!selectedFile) {
        alert('Пожалуйста, выберите файл!');
        return;
    }
    
    if (!selectedFilter) {
        alert('Пожалуйста, выберите фильтр!');
        return;
    }
    
    // Показываем индикатор загрузки
    document.getElementById('loadingOverlay').style.display = 'flex';
    
    // Формируем данные для отправки
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('filter', selectedFilter.id);
    
    // Добавляем параметры фильтра
    if (selectedFilter.params) {
        selectedFilter.params.forEach(param => {
            const value = document.getElementById(param.name).value;
            formData.append(param.name, value);
        });
    }
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showResult(result);
        } else {
            alert('Ошибка обработки: ' + result.error);
        }
    } catch (error) {
        alert('Ошибка сервера: ' + error.message);
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
}

// Показать результат
function showResult(result) {
    outputFilename = result.output_file;
    
    // Обновляем статистику
    document.getElementById('execTime').textContent = result.execution_time + ' сек';
    document.getElementById('fileSize').textContent = result.file_size + ' КБ';
    document.getElementById('filterUsed').textContent = selectedFilter.name;
    
    // Показываем изображения до/после
    const beforeImg = document.getElementById('beforeImage');
    const afterImg = document.getElementById('afterImage');
    
    beforeImg.src = document.getElementById('previewImage').src;
    afterImg.src = '/view/' + outputFilename;
    
    // Настраиваем кнопку скачивания
    document.getElementById('downloadBtn').onclick = () => {
        window.location.href = '/download/' + outputFilename;
    };
    
    // Скрываем предыдущие секции и показываем результат
    document.getElementById('uploadBox').style.display = 'none';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('filterSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'block';
    
    // Прокручиваем к результату
    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
}

// Обработать другое изображение
function processAnother() {
    resetForm();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Сброс формы
function resetForm() {
    selectedFile = null;
    selectedFilter = null;
    outputFilename = null;
    
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadBox').style.display = 'block';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('filterSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    
    document.querySelectorAll('.filter-card').forEach(card => {
        card.classList.remove('selected');
    });
}

