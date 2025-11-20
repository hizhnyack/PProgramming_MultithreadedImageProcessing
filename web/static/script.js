// Глобальные переменные
let selectedFile = null;
let selectedFiles = [];  // Массив выбранных файлов
let selectedFilter = null;
let filters = [];
let outputFilename = null;
let batchProcessing = false;  // Флаг пакетной обработки

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
                        <span>0°</span>
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
        handleFiles(Array.from(files));
    }
}

// Настройка выбора файла
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFiles(Array.from(this.files));
        }
    });
}

// Обработка выбранных файлов
function handleFiles(files) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
    const maxSizePerFile = 16 * 1024 * 1024; // 16 МБ на файл
    const maxTotalSize = 3 * 1024 * 1024 * 1024; // 3 ГБ общий размер запроса
    
    // Фильтруем файлы
    const validFiles = [];
    let totalSize = 0;
    
    for (const file of files) {
        if (!allowedTypes.includes(file.type)) {
            alert(`Файл "${file.name}" имеет недопустимый формат. Используйте PNG, JPG или BMP.`);
            continue;
        }
        
        if (file.size > maxSizePerFile) {
            alert(`Файл "${file.name}" слишком большой. Максимальный размер одного файла: 16 МБ.`);
            continue;
        }
        
        // Проверяем общий размер
        if (totalSize + file.size > maxTotalSize) {
            alert(`Общий размер файлов превышает лимит (3 ГБ). Файл "${file.name}" не будет добавлен.`);
            continue;
        }
        
        totalSize += file.size;
        validFiles.push(file);
    }
    
    if (validFiles.length === 0) {
        return;
    }
    
    // Если один файл - используем старый режим
    if (validFiles.length === 1) {
        selectedFile = validFiles[0];
        selectedFiles = [];
        batchProcessing = false;
        
        // Показываем предпросмотр
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('fileName').textContent = validFiles[0].name;
            document.getElementById('previewSection').style.display = 'block';
            document.getElementById('filesList').style.display = 'none';
            document.getElementById('filterSection').style.display = 'block';
        };
        reader.readAsDataURL(validFiles[0]);
    } else {
        // Несколько файлов - пакетная обработка
        selectedFile = null;
        selectedFiles = validFiles;
        batchProcessing = true;
        
        // Скрываем предпросмотр одного файла
        document.getElementById('previewSection').style.display = 'none';
        
        // Показываем список файлов
        renderFilesList();
        document.getElementById('filesList').style.display = 'block';
        document.getElementById('filterSection').style.display = 'block';
    }
}

// Отрисовка списка файлов
function renderFilesList() {
    const filesGrid = document.getElementById('filesGrid');
    const filesCount = document.getElementById('filesCount');
    
    filesGrid.innerHTML = '';
    
    // Вычисляем общий размер
    let totalSize = 0;
    selectedFiles.forEach(file => {
        totalSize += file.size;
    });
    
    // Отображаем количество и общий размер
    const totalSizeGB = (totalSize / (1024 * 1024 * 1024)).toFixed(2);
    const maxSizeGB = 3;
    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(0);
    filesCount.textContent = `${selectedFiles.length} файл(ов) (${totalSizeGB} ГБ / ${maxSizeGB} ГБ)`;
    
    // Предупреждение если размер близок к лимиту
    if (totalSize > maxSizeGB * 1024 * 1024 * 1024 * 0.9) {
        filesCount.style.color = '#dc3545';
        filesCount.textContent += ' ⚠️';
    } else {
        filesCount.style.color = '#333';
    }
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.dataset.index = index;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const fileSizeKB = (file.size / 1024).toFixed(1);
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            const sizeText = file.size > 1024 * 1024 ? `${fileSizeMB} МБ` : `${fileSizeKB} КБ`;
            
            fileItem.innerHTML = `
                <button class="remove-btn" onclick="removeFile(${index})" title="Удалить">×</button>
                <img src="${e.target.result}" alt="${file.name}">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${sizeText}</div>
            `;
        };
        reader.readAsDataURL(file);
        
        filesGrid.appendChild(fileItem);
    });
}

// Удаление файла из списка
function removeFile(index) {
    selectedFiles.splice(index, 1);
    
    if (selectedFiles.length === 0) {
        clearFiles();
    } else {
        renderFilesList();
    }
}

// Очистка списка файлов
function clearFiles() {
    selectedFiles = [];
    selectedFile = null;
    document.getElementById('filesList').style.display = 'none';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('filterSection').style.display = 'none';
    document.getElementById('fileInput').value = '';
}

// Обработка изображения
async function processImage() {
    if (!selectedFile && selectedFiles.length === 0) {
        alert('Пожалуйста, выберите файл(ы)!');
        return;
    }
    
    if (!selectedFilter) {
        alert('Пожалуйста, выберите фильтр!');
        return;
    }
    
    // Если один файл - используем старый режим
    if (!batchProcessing && selectedFile) {
        await processSingleFile();
    } else if (batchProcessing && selectedFiles.length > 0) {
        await processBatchFiles();
    }
}

// Обработка одного файла
async function processSingleFile() {
    // Показываем индикатор загрузки
    document.getElementById('loadingOverlay').style.display = 'flex';
    
    // Получаем выбранный режим обработки
    const processorMode = document.querySelector('input[name="processor"]:checked').value;
    
    // Формируем данные для отправки
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('filter', selectedFilter.id);
    formData.append('processor', processorMode);
    
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
        
        // Проверяем статус ответа
        if (!response.ok) {
            let errorMessage = 'Ошибка сервера: ' + response.statusText;
            try {
                const errorData = await response.json();
                if (errorData.error) {
                    errorMessage = errorData.error;
                }
            } catch (e) {
                if (response.status === 413) {
                    errorMessage = 'Размер файла слишком большой! Максимальный размер: 16 МБ.';
                }
            }
            alert(errorMessage);
            return;
        }
        
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

// Пакетная обработка файлов
async function processBatchFiles() {
    // Скрываем секции и показываем результаты пакетной обработки
    document.getElementById('uploadBox').style.display = 'none';
    document.getElementById('filesList').style.display = 'none';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('filterSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('batchResultSection').style.display = 'block';
    
    // Сбрасываем статистику
    document.getElementById('batchSuccess').textContent = '0';
    document.getElementById('batchFailed').textContent = '0';
    document.getElementById('batchTime').textContent = '-';
    document.getElementById('resultsList').innerHTML = '';
    
    // Обновляем прогресс
    updateProgress(0, 'Начало обработки...');
    
    // Получаем выбранный режим обработки
    const processorMode = document.querySelector('input[name="processor"]:checked').value;
    
    // Формируем данные для отправки
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    formData.append('filter', selectedFilter.id);
    formData.append('processor', processorMode);
    
    // Добавляем параметры фильтра
    if (selectedFilter.params) {
        selectedFilter.params.forEach(param => {
            const value = document.getElementById(param.name).value;
            formData.append(param.name, value);
        });
    }
    
    const startTime = Date.now();
    
    try {
        const response = await fetch('/upload_batch', {
            method: 'POST',
            body: formData
        });
        
        // Проверяем статус ответа
        if (!response.ok) {
            // Пытаемся получить JSON с описанием ошибки
            let errorMessage = 'Ошибка сервера: ' + response.statusText;
            try {
                const errorData = await response.json();
                if (errorData.error) {
                    errorMessage = errorData.error;
                }
            } catch (e) {
                // Если не удалось распарсить JSON, используем стандартное сообщение
                if (response.status === 413) {
                    errorMessage = 'Размер загружаемых файлов слишком большой! Максимальный размер запроса: 3 ГБ.';
                }
            }
            alert(errorMessage);
            return;
        }
        
        // Используем Server-Sent Events или polling для прогресса
        // Для простоты используем polling
        const result = await response.json();
        
        if (result.success) {
            const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
            showBatchResults(result, totalTime);
        } else {
            alert('Ошибка обработки: ' + result.error);
        }
    } catch (error) {
        alert('Ошибка сервера: ' + error.message);
    }
}

// Обновление прогресса
function updateProgress(percent, text) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    progressBar.style.width = percent + '%';
    progressBar.textContent = percent.toFixed(0) + '%';
    progressText.textContent = text;
}

// Показ результатов пакетной обработки
function showBatchResults(result, totalTime) {
    document.getElementById('batchSuccess').textContent = result.success_count || 0;
    document.getElementById('batchFailed').textContent = result.failed_count || 0;
    document.getElementById('batchTime').textContent = totalTime + ' сек';
    
    updateProgress(100, 'Обработка завершена!');
    
    const resultsList = document.getElementById('resultsList');
    resultsList.innerHTML = '';
    
    if (result.results && result.results.length > 0) {
        result.results.forEach((item, index) => {
            const resultItem = document.createElement('div');
            resultItem.className = `result-item ${item.success ? 'success' : 'error'}`;
            
            let content = `
                <div class="result-status">${item.success ? '✅' : '❌'}</div>
                <div class="result-name">${item.filename}</div>
            `;
            
            if (item.success && item.output_file) {
                content += `
                    <img src="/view/${item.output_file}" alt="Result">
                    <div class="result-time">${item.execution_time ? item.execution_time.toFixed(3) + ' сек' : '-'}</div>
                    <button class="download-btn" onclick="window.location.href='/download/${item.output_file}'">
                        💾 Скачать
                    </button>
                `;
            } else {
                content += `
                    <div class="result-time" style="color: #dc3545;">${item.error || 'Ошибка обработки'}</div>
                `;
            }
            
            resultItem.innerHTML = content;
            resultsList.appendChild(resultItem);
        });
    }
    
    // Прокручиваем к результатам
    document.getElementById('batchResultSection').scrollIntoView({ behavior: 'smooth' });
}

// Показать результат
function showResult(result) {
    outputFilename = result.output_file;
    
    // Получаем выбранный режим обработки
    const processorMode = document.querySelector('input[name="processor"]:checked').value;
    const processorLabel = processorMode === 'gpu' ? '🚀 GPU (CUDA)' : '💻 CPU';
    
    // Обновляем статистику
    document.getElementById('execTime').textContent = result.execution_time + ' сек';
    document.getElementById('fileSize').textContent = result.file_size + ' КБ';
    document.getElementById('filterUsed').textContent = selectedFilter.name + ' [' + processorLabel + ']';
    
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
    selectedFiles = [];
    selectedFilter = null;
    outputFilename = null;
    batchProcessing = false;
    
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadBox').style.display = 'block';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('filesList').style.display = 'none';
    document.getElementById('filterSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('batchResultSection').style.display = 'none';
    
    document.querySelectorAll('.filter-card').forEach(card => {
        card.classList.remove('selected');
    });
}

