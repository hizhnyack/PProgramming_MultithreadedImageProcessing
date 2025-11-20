// Глобальные переменные
let selectedFiles = [];  // Массив файлов
let selectedFilter = null;
let filters = [];
let outputFilenames = [];  // Массив результатов

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
            
            let html = '';
            
            // Обработка разных типов параметров
            if (param.type === 'buttons') {
                // Группа кнопок
                html = `<label>${param.label}</label>
                        <div class="button-group" id="${param.name}_group">`;
                param.options.forEach(option => {
                    const active = option.value === param.default ? 'active' : '';
                    html += `
                        <button type="button" 
                                class="param-button ${active}" 
                                data-value="${option.value}"
                                onclick="selectButton('${param.name}', '${option.value}')">
                            <span class="button-icon">${option.icon}</span>
                            <span class="button-label">${option.label}</span>
                        </button>
                    `;
                });
                html += `</div>
                         <input type="hidden" id="${param.name}" name="${param.name}" value="${param.default}">`;
            } else if (param.type === 'select') {
                // Выпадающий список
                html = `
                    <label>${param.label}</label>
                    <select id="${param.name}" name="${param.name}" class="param-select" onchange="handleParamChange('${param.name}')">
                `;
                param.options.forEach(option => {
                    const selected = option.value === param.default ? 'selected' : '';
                    html += `<option value="${option.value}" ${selected}>${option.label}</option>`;
                });
                html += `</select>`;
            } else if (param.type === 'range') {
                // Ползунок
                html = `
                    <label>
                        ${param.label}
                        <span class="param-value" id="${param.name}_value">${param.default}${unit}</span>
                    </label>
                    <input 
                        type="range" 
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
            }
            
            // Добавляем атрибуты для условного отображения
            if (param.depends_on) {
                group.dataset.dependsOn = param.depends_on;
                group.dataset.dependsValue = param.depends_value;
                group.style.display = 'none';  // Скрываем по умолчанию
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

// Настройка выбора файлов
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFiles(Array.from(this.files));
        }
    });
}

// Обработка нескольких файлов
function handleFiles(files) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
    selectedFiles = [];
    
    // Фильтруем и проверяем файлы
    for (const file of files) {
        if (!allowedTypes.includes(file.type)) {
            alert(`Файл ${file.name} имеет недопустимый формат! Используйте PNG, JPG или BMP.`);
            continue;
        }
        
        if (file.size > 16 * 1024 * 1024) {
            alert(`Файл ${file.name} слишком большой! Максимальный размер: 16 МБ.`);
            continue;
        }
        
        selectedFiles.push(file);
    }
    
    if (selectedFiles.length === 0) {
        return;
    }
    
    // Показываем предпросмотр
    const previewGrid = document.getElementById('previewGrid');
    previewGrid.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            previewItem.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <p class="preview-filename">${file.name}</p>
                <button class="btn-remove" onclick="removeFile(${index})">✕</button>
            `;
            previewGrid.appendChild(previewItem);
        };
        reader.readAsDataURL(file);
    });
    
    document.getElementById('fileCount').textContent = selectedFiles.length;
    document.getElementById('previewSection').style.display = 'block';
    document.getElementById('filterSection').style.display = 'block';
}

// Удалить файл из списка
function removeFile(index) {
    selectedFiles.splice(index, 1);
    if (selectedFiles.length === 0) {
        resetForm();
    } else {
        handleFiles(selectedFiles);
    }
}

// Обработка изображений
async function processImage() {
    if (selectedFiles.length === 0) {
        alert('Пожалуйста, выберите файлы!');
        return;
    }
    
    if (!selectedFilter) {
        alert('Пожалуйста, выберите фильтр!');
        return;
    }
    
    // Показываем индикатор загрузки и прогресс
    document.getElementById('loadingOverlay').style.display = 'flex';
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (selectedFiles.length > 1) {
        progressContainer.style.display = 'block';
        document.getElementById('loadingText').textContent = 'Обработка изображений на GPU...';
    }
    
    outputFilenames = [];
    let processed = 0;
    
    // Обрабатываем каждый файл
    for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('file', file);
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
                outputFilenames.push({
                    original: file.name,
                    output: result.output_file,
                    time: result.execution_time,  // Исправлено: execution_time вместо processing_time
                    size: result.file_size        // Исправлено: file_size вместо output_size
                });
            } else {
                console.error(`Ошибка обработки ${file.name}: ${result.error}`);
            }
        } catch (error) {
            console.error(`Ошибка сервера для ${file.name}: ${error.message}`);
        }
        
        processed++;
        progressBar.style.width = `${(processed / selectedFiles.length) * 100}%`;
        progressText.textContent = `${processed} / ${selectedFiles.length}`;
    }
    
    document.getElementById('loadingOverlay').style.display = 'none';
    
    if (outputFilenames.length > 0) {
        showResults();
    } else {
        alert('Не удалось обработать ни одного изображения!');
    }
}

// Показать результаты
function showResults() {
    // Скрываем предыдущие секции и показываем результат
    document.getElementById('uploadBox').style.display = 'none';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('filterSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'block';
    
    // Обновляем статистику
    const totalTime = outputFilenames.reduce((sum, r) => sum + parseFloat(r.time), 0).toFixed(3);
    const totalSize = outputFilenames.reduce((sum, r) => sum + parseFloat(r.size), 0).toFixed(2);
    
    document.getElementById('execTime').textContent = totalTime + ' сек';
    document.getElementById('fileSize').textContent = totalSize + ' КБ';
    document.getElementById('filterUsed').textContent = `${selectedFilter.name} (${outputFilenames.length} файлов)`;
    
    // Показываем первое изображение в сравнении
    if (outputFilenames.length > 0) {
        const beforeImg = document.getElementById('beforeImage');
        const afterImg = document.getElementById('afterImage');
        
        const reader = new FileReader();
        reader.onload = function(e) {
            beforeImg.src = e.target.result;
        };
        reader.readAsDataURL(selectedFiles[0]);
        
        afterImg.src = '/view/' + outputFilenames[0].output;
        
        // Настраиваем кнопки скачивания
        const downloadBtn = document.getElementById('downloadBtn');
        const downloadAllBtn = document.getElementById('downloadAllBtn');
        const filesList = document.getElementById('filesList');
        const filesListContent = document.getElementById('filesListContent');
        
        if (outputFilenames.length === 1) {
            // Один файл - обычная кнопка скачивания
            downloadBtn.textContent = '💾 Скачать';
            downloadBtn.onclick = () => {
                window.location.href = '/download/' + outputFilenames[0].output;
            };
            downloadAllBtn.style.display = 'none';
            filesList.style.display = 'none';
        } else {
            // Несколько файлов - показываем обе кнопки и список
            downloadBtn.textContent = '💾 Скачать первый';
            downloadBtn.onclick = () => {
                window.location.href = '/download/' + outputFilenames[0].output;
            };
            downloadAllBtn.style.display = 'inline-block';
            filesList.style.display = 'block';
            
            // Создаем список файлов с индивидуальными кнопками
            filesListContent.innerHTML = '';
            outputFilenames.forEach((result, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <span class="file-number">${index + 1}.</span>
                    <span class="file-name">${result.original}</span>
                    <span class="file-stats">⏱️ ${result.time} сек | 📦 ${result.size} КБ</span>
                    <button class="btn btn-sm btn-primary" onclick="downloadSingleFile('${result.output}', '${result.original}')">
                        💾 Скачать
                    </button>
                `;
                filesListContent.appendChild(fileItem);
            });
            
            // Показываем галерею всех изображений
            const resultsGallery = document.getElementById('resultsGallery');
            const galleryGrid = document.getElementById('galleryGrid');
            resultsGallery.style.display = 'block';
            galleryGrid.innerHTML = '';
            
            outputFilenames.forEach((result, index) => {
                const galleryItem = document.createElement('div');
                galleryItem.className = 'gallery-item';
                
                // Создаем превью для оригинального изображения
                const reader = new FileReader();
                reader.onload = function(e) {
                    galleryItem.innerHTML = `
                        <h4>${result.original}</h4>
                        <div class="gallery-comparison">
                            <div class="gallery-image-box">
                                <p>До</p>
                                <img src="${e.target.result}" alt="Before">
                            </div>
                            <div class="gallery-image-box">
                                <p>После</p>
                                <img src="/view/${result.output}?t=${Date.now()}" alt="After">
                            </div>
                        </div>
                        <div class="gallery-stats">
                            <span>⏱️ ${result.time} сек</span>
                            <span>📦 ${result.size} КБ</span>
                        </div>
                        <button class="btn btn-sm btn-primary" style="width: 100%;" onclick="downloadSingleFile('${result.output}', '${result.original}')">
                            💾 Скачать
                        </button>
                    `;
                };
                reader.readAsDataURL(selectedFiles[index]);
                
                galleryGrid.appendChild(galleryItem);
            });
        }
    }
    
    // Прокручиваем к результату
    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
}

// Скачать все результаты в ZIP
async function downloadAllResults() {
    try {
        const filenames = outputFilenames.map(r => r.output);
        
        const response = await fetch('/download_all', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ files: filenames })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `processed_images_${Date.now()}.zip`;
            link.click();
            window.URL.revokeObjectURL(url);
        } else {
            alert('Ошибка при скачивании файлов');
        }
    } catch (error) {
        console.error('Ошибка:', error);
        alert('Ошибка при скачивании: ' + error.message);
    }
}

// Скачать один файл
function downloadSingleFile(outputFile, originalName) {
    const link = document.createElement('a');
    link.href = '/download/' + outputFile;
    link.download = `processed_${originalName}`;
    link.click();
}

// Обработать другое изображение
function processAnother() {
    resetForm();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Выбор кнопки в группе
function selectButton(paramName, value) {
    // Убираем active у всех кнопок в группе
    const group = document.getElementById(paramName + '_group');
    group.querySelectorAll('.param-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Добавляем active выбранной кнопке
    const selectedBtn = group.querySelector(`[data-value="${value}"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
    }
    
    // Обновляем скрытое поле
    document.getElementById(paramName).value = value;
    
    // Вызываем обработчик изменения
    handleParamChange(paramName);
}

// Обработка изменения параметров (для условного отображения)
function handleParamChange(paramName) {
    const value = document.getElementById(paramName).value;
    
    // Показываем/скрываем зависимые параметры
    document.querySelectorAll('.param-group[data-depends-on]').forEach(group => {
        if (group.dataset.dependsOn === paramName) {
            if (group.dataset.dependsValue === value) {
                group.style.display = 'block';
            } else {
                group.style.display = 'none';
            }
        }
    });
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

