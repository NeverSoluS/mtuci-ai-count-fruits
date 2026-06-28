console.log('[JS] Загрузка script.js');

var MODE_NAMES = {
    detection: '🔍 Детекция',
    segmentation: '🎭 Сегментация',
    classification: '🏷️ Классификация',
};

var statsTimer = null;
var cameraRunning = false;
var currentMode = null;
var selectedCameraIndex = 0;
var availableCameras = [];
var peerConnection = null;

// Включаем плавные переходы для всех элементов с классом .fading
var cameraListLoaded = false;

document.querySelectorAll('.tab[data-tab]').forEach(function(btn) {
    btn.addEventListener('click', function() {
        var name = btn.getAttribute('data-tab');
        document.querySelectorAll('.tabs > .tab[data-tab]').forEach(function(t) {
            t.classList.remove('active');
        });
        document.querySelectorAll('.panel').forEach(function(p) {
            p.classList.remove('active');
        });
        btn.classList.add('active');
        document.getElementById('panel-' + name).classList.add('active');

        if (name === 'history') {
            loadHistory();
        }
        if (name === 'stream' && !cameraListLoaded) {
            loadCameraList();
        }
    });
});


// Функция переключения (вызывается по кнопке)
function toggleTheme() {
    var html = document.documentElement;
    var currentTheme = html.getAttribute('data-theme');
    var newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    // Применяем тему
    html.setAttribute('data-theme', newTheme);
    
    // Сохраняем в localStorage
    try {
        localStorage.setItem('theme', newTheme);
    } catch (e) {
        // На случай если localStorage отключен
    }
    
    // Обновляем иконку
    updateThemeIcon(newTheme);
}

// Обновление иконки кнопки
function updateThemeIcon(theme) {
    var icon = document.getElementById('themeIcon');
    if (icon) {
        // Плавная анимация смены иконки
        icon.style.transform = 'rotate(180deg) scale(0)';
        setTimeout(function() {
            icon.textContent = theme === 'light' ? '🌙' : '☀️';
            icon.style.transform = 'rotate(0deg) scale(1)';
        }, 150);
    }
}

// Инициализация иконки при загрузке (тема уже установлена inline-скриптом)
(function() {
    var theme = document.documentElement.getAttribute('data-theme') || 'light';
    
    // Ждём появления DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            updateThemeIcon(theme);
        });
    } else {
        updateThemeIcon(theme);
    }
})();

// Слушаем изменения системной темы (если пользователь не выбрал свою)
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        // Меняем только если пользователь сам не выбирал тему
        if (!localStorage.getItem('theme')) {
            var newTheme = e.matches ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            updateThemeIcon(newTheme);
        }
    });
}

// Делаем toggleTheme глобальной (для onclick)
window.toggleTheme = toggleTheme;

// Загрузка сохранённой темы
(function() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Обновляем иконку после загрузки DOM
    window.addEventListener('DOMContentLoaded', function() {
        const icon = document.getElementById('themeIcon');
        if (icon) {
            icon.textContent = savedTheme === 'light' ? '🌙' : '☀️';
        }
    });
})();

document.querySelectorAll('.tab[data-cam]').forEach(function(btn) {
    btn.addEventListener('click', function() {
        var mode = btn.getAttribute('data-cam');
        document.querySelectorAll('.tab[data-cam]').forEach(function(t) {
            t.classList.remove('active');
        });
        btn.classList.add('active');
        document.getElementById('cam-webcam').style.display = mode === 'webcam' ? 'block' : 'none';
        document.getElementById('cam-rtsp').style.display = mode === 'rtsp' ? 'block' : 'none';
    });
});

// функция для отправки файла на сервер и получения результата
document.getElementById('runBtn').addEventListener('click', async function() {
    var file = document.getElementById('fileInput').files[0];
    if (!file) { alert('Выберите файл'); return; }

    var fd = new FormData();
    fd.append('file', file);
    var modeRadio = document.querySelector('input[name="mode"]:checked');
    fd.append('mode', modeRadio ? modeRadio.value : 'detection');
    fd.append('engine', modeRadio ? (modeRadio.getAttribute('data-engine') || 'yolo') : 'yolo');

    var btn = document.getElementById('runBtn');
    var div = document.getElementById('fileResult');
    btn.disabled = true;
    btn.textContent = 'Обработка...';
    div.innerHTML = '<p style="color:#888;">⏳ Обработка...</p>';

    try {
        var response = await fetch('/process', {method: 'POST', body: fd});
        var d = await response.json();
        if (!response.ok) throw new Error(d.error || ('Ошибка ' + response.status));

        var html = '<h4>✅ Результат</h4>';
        html += '<div class="debug-info">';
        html += '<p><b>Режим:</b> ' + MODE_NAMES[d.mode] + '</p>';
        html += '<p><b>Время:</b> ' + d.processing_time + ' сек</p>';

        if (d.mode === 'classification' && d.top5) {
            html += '<p><b>Top-5 предсказаний:</b></p><ul>';
            for (var k in d.top5) {
                html += '<li>' + k + ': <b>' + (d.top5[k]*100).toFixed(1) + '%</b></li>';
            }
            html += '</ul>';
        } else {
            html += '<p><b>Всего фруктов:</b> ' + d.count + '</p>';
            var engineName = d.engine === 'owlvit' ? '🎯 OWL-ViT (точный)' : '⚡ YOLOv8m (быстрый)';
            html += '<p><b>Движок:</b> ' + engineName + '</p>';
            if (d.by_class && Object.keys(d.by_class).length > 0) {
                html += '<table><tr><th>Фрукт</th><th>Количество</th></tr>';
                for (var k in d.by_class) {
                    html += '<tr><td>' + k + '</td><td><b>' + d.by_class[k] + '</b></td></tr>';
                }
                html += '</table>';
            }
        }

        html += '</div>';

        var url = '/' + d.result_url + '?t=' + Date.now();
        if (d.is_video) {
            html += '<video controls src="' + url + '" style="margin-top:15px; max-width:100%;"></video>';
        } else {
            html += '<img src="' + url + '" style="margin-top:15px; max-width:100%;">';
        }

        div.innerHTML = html;
    } catch (e) {
        div.innerHTML = '<div class="status error">❌ ' + e.message + '</div>';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Запустить обработку';
    }
});

// Список камер с плавной анимацией и без перерисовки всего списка при выборе
async function loadCameraList() {
    var div = document.getElementById('cameraList');
    var refreshBtn = document.getElementById('refreshCamerasBtn');
    
    // Плавное затухание перед обновлением
    div.classList.add('fading');
    if (refreshBtn) refreshBtn.classList.add('refreshing');
    
    // Ждём окончания анимации затухания
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Показываем индикатор загрузки
    div.innerHTML = '<div class="camera-loading">' +
                    '<div class="spinner"></div>' +
                    '<span>Поиск камер...</span>' +
                    '</div>';
    div.classList.remove('fading');
    
    try {
        // Временный доступ для получения label устройств
        var tempStream = null;
        try {
            tempStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        } catch (e) {
            console.warn('Нет доступа к камере:', e);
            // Плавное появление сообщения об ошибке
            await fadeOut(div);
            div.innerHTML = '<p style="color:var(--accent-warning); padding:10px;">' +
                '⚠️️ Разрешите доступ к камере в браузере, чтобы видеть названия устройств.</p>';
            await fadeIn(div);
            if (refreshBtn) refreshBtn.classList.remove('refreshing');
            return;
        }

        var devices = await navigator.mediaDevices.enumerateDevices();
        var cameras = devices.filter(function(d) { return d.kind === 'videoinput'; });

        if (tempStream) {
            tempStream.getTracks().forEach(function(t) { t.stop(); });
        }

        if (cameras.length === 0) {
            await fadeOut(div);
            div.innerHTML = '<p style="color:var(--accent-danger); padding:10px;">❌ Камеры не найдены.</p>';
            await fadeIn(div);
            if (refreshBtn) refreshBtn.classList.remove('refreshing');
            return;
        }

        availableCameras = cameras;
        cameraListLoaded = true;

        // Плавное затухание перед показом списка
        await fadeOut(div);
        
        var html = '<p style="color:var(--text-secondary); margin-bottom:12px; font-weight:600;">' +
                   'Доступные камеры (' + cameras.length + '):</p>';
        
        cameras.forEach(function(cam, i) {
            var label = cam.label || ('Камера ' + i);
            var isSelected = (i === selectedCameraIndex);
            var className = isSelected ? 'camera-option selected' : 'camera-option';
            
            html += '<label class="' + className + '">' +
                    '<input type="radio" name="camera" value="' + i + '" ' +
                    (isSelected ? 'checked' : '') + '> ' +
                    '<span class="camera-label">' + label + '</span>' +
                    '</label>';
        });

        div.innerHTML = html;
        
        // Плавное появление
        await fadeIn(div);
        
        // Снимаем индикатор загрузки с кнопки
        if (refreshBtn) refreshBtn.classList.remove('refreshing');

        // Обработчики выбора (БЕЗ перерисовки всего списка!)
        document.querySelectorAll('input[name="camera"]').forEach(function(radio) {
            radio.addEventListener('change', function() {
                var newIndex = parseInt(radio.value);
                if (newIndex !== selectedCameraIndex) {
                    selectedCameraIndex = newIndex;
                    // Только обновляем стили, без перерисовки DOM
                    updateCameraSelection();
                }
            });
        });

    } catch (e) {
        await fadeOut(div);
        div.innerHTML = '<p style="color:var(--accent-danger); padding:10px;">❌ Ошибка: ' + e.message + '</p>';
        await fadeIn(div);
        if (refreshBtn) refreshBtn.classList.remove('refreshing');
    }
}

// Вспомогательные функции для плавных переходов

function fadeOut(element) {
    return new Promise(resolve => {
        element.classList.add('fading');
        setTimeout(resolve, 200);
    });
}

function fadeIn(element) {
    return new Promise(resolve => {
        element.classList.remove('fading');
        setTimeout(resolve, 250);
    });
}

// Обновление выделения камеры БЕЗ перерисовки всего списка
function updateCameraSelection() {
    document.querySelectorAll('.camera-option').forEach(function(label, i) {
        var radio = label.querySelector('input[type="radio"]');
        if (i === selectedCameraIndex) {
            label.classList.add('selected');
            if (radio) radio.checked = true;
        } else {
            label.classList.remove('selected');
            if (radio) radio.checked = false;
        }
    });
}

// Делаем loadCameraList глобальной
window.loadCameraList = loadCameraList;

// Включение/выключение камеры по кнопке
document.getElementById('camToggleBtn').addEventListener('click', async function() {
    if (cameraRunning && currentMode === 'webcam') {
        await stopCamera();
    } else {
        await startCamera();
    }
});

// Обновление информации о движке при смене режима 
var isUpdatingEngine = false;

function updateEngineInfo() {
    if (isUpdatingEngine) return;  // Защита от двойных кликов
    
    var radio = document.querySelector('input[name="mode"]:checked');
    var info = document.getElementById('engineInfo');
    if (!radio || !info) return;
    
    var mode = radio.value;
    var engine = radio.getAttribute('data-engine');
    
    isUpdatingEngine = true;
    
    // Плавное исчезновение (fade-out)
    info.classList.add('fading');
    info.classList.remove('appearing', 'pulse');
    
    setTimeout(function() {
        // Обновляем контент
        var newContent = '';
        var newColor = 'var(--accent-primary)';
        var indicatorClass = 'fast';
        
        if (mode === 'detection' && engine === 'owlvit') {
            newContent = '<span class="engine-indicator accurate"></span>' +
                '<b>🎯 Режим: OWL-ViT (точный, zero-shot)</b><br>' +
                '<small>Распознаёт <b>70+ фруктов</b>: яблоко, банан, апельсин, ананас, ' +
                'виноград, манго, клубника, арбуз, персик и др.<br>' +
                '⏱️ Скорость: 3-10 секунд на фото. Точность: ~60%.</small>';
            newColor = 'var(--accent-warning)';
            indicatorClass = 'accurate';
        } else if (mode === 'detection') {
            newContent = '<span class="engine-indicator fast"></span>' +
                '<b>⚡ Режим: YOLOv8m (быстрый)</b><br>' +
                '<small>Распознаёт: 🍎 Яблоко, 🍌 Банан, 🍊 Апельсин. ' +
                '⏱️ Скорость: &lt;1 сек. Точность: ~85%.</small>';
            newColor = 'var(--accent-primary)';
            indicatorClass = 'fast';
        } else if (mode === 'segmentation') {
            newContent = '<span class="engine-indicator segment"></span>' +
                '<b>🎭 Режим: Сегментация (YOLOv8m-seg)</b><br>' +
                '<small>Выделяет <b>маски/контуры</b> фруктов. 3 вида фруктов.<br>' +
                '⏱️ Скорость: 1-2 сек на фото.</small>';
            newColor = 'var(--accent-success)';
            indicatorClass = 'segment';
        } else if (mode === 'classification') {
            newContent = '<span class="engine-indicator classify"></span>' +
                '<b>🏷️ Режим: Классификация</b><br>' +
                '<small>Показывает <b>Top-5</b> предсказаний с процентами.<br>' +
                '⏱️ Скорость: &lt;1 сек на фото.</small>';
            newColor = 'var(--accent-secondary)';
            indicatorClass = 'classify';
        }
        
        // Обновляем HTML
        info.innerHTML = newContent;
        info.style.borderLeftColor = newColor;
        
        // Плавное появление (fade-in)
        info.classList.remove('fading');
        info.classList.add('appearing');
        
        // Shimmer эффект
        info.classList.add('updating');
        setTimeout(function() {
            info.classList.remove('updating');
        }, 800);
        
        // Пульсация
        info.classList.add('pulse');
        
        // Разблокируем через 350мс
        setTimeout(function() {
            info.classList.remove('appearing');
            isUpdatingEngine = false;
        }, 350);
        
    }, 250);  // Ждём окончания fade-out
}

// Привязываем обработчик к radio-кнопкам режима
document.querySelectorAll('input[name="mode"]').forEach(function(radio) {
    radio.addEventListener('change', updateEngineInfo);
});

// Инициализация при загрузке (без анимации)
window.addEventListener('DOMContentLoaded', function() {
    var radio = document.querySelector('input[name="mode"]:checked');
    var info = document.getElementById('engineInfo');
    if (!radio || !info) return;
    
    var mode = radio.value;
    var engine = radio.getAttribute('data-engine');
    
    var newContent = '';
    var newColor = 'var(--accent-primary)';
    
    if (mode === 'detection' && engine === 'owlvit') {
        newContent = '<span class="engine-indicator accurate"></span>' +
            '<b>🎯 Режим: OWL-ViT (точный, zero-shot)</b><br>' +
            '<small>Распознаёт <b>70+ фруктов</b>: яблоко, банан, апельсин, ананас, ' +
            'виноград, манго, клубника, арбуз, персик и др.<br>' +
            '⏱️ Скорость: 3-10 секунд на фото. Точность: ~60%.</small>';
        newColor = 'var(--accent-warning)';
    } else if (mode === 'detection') {
        newContent = '<span class="engine-indicator fast"></span>' +
            '<b>⚡ Режим: YOLOv8m (быстрый)</b><br>' +
            '<small>Распознаёт: 🍎 Яблоко, 🍌 Банан, 🍊 Апельсин. ' +
            '⏱️ Скорость: &lt;1 сек. Точность: ~85%.</small>';
        newColor = 'var(--accent-primary)';
    } else if (mode === 'segmentation') {
        newContent = '<span class="engine-indicator segment"></span>' +
            '<b>🎭 Режим: Сегментация (YOLOv8m-seg)</b><br>' +
            '<small>Выделяет <b>маски/контуры</b> фруктов. 3 вида фруктов.<br>' +
            '⏱️ Скорость: 1-2 сек на фото.</small>';
        newColor = 'var(--accent-success)';
    } else if (mode === 'classification') {
        newContent = '<span class="engine-indicator classify"></span>' +
            '<b>🏷️ Режим: Классификация</b><br>' +
            '<small>Показывает <b>Top-5</b> предсказаний с процентами.<br>' +
            '⏱️ Скорость: &lt;1 сек на фото.</small>';
        newColor = 'var(--accent-secondary)';
    }
    
    info.innerHTML = newContent;
    info.style.borderLeftColor = newColor;
});

// Привязываем обработчик к radio-кнопкам режима
document.querySelectorAll('input[name="mode"]').forEach(function(radio) {
    radio.addEventListener('change', updateEngineInfo);
});

// Инициализация при загрузке
window.addEventListener('DOMContentLoaded', updateEngineInfo);

async function startCamera() {
    var btn = document.getElementById('camToggleBtn');
    var status = document.getElementById('webcamStatus');
    var localImg = document.getElementById('localImg');
    var remoteImg = document.getElementById('remoteImg');

    btn.disabled = true;
    status.style.display = 'block';

    var camLabel = availableCameras[selectedCameraIndex]
        ? (availableCameras[selectedCameraIndex].label || ('Камера ' + selectedCameraIndex))
        : 'Камера';
    status.textContent = '🔌 Запуск "' + camLabel + '"...';
    status.className = 'status warning';

    try {
        var r = await fetch('/mjpeg/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({index: selectedCameraIndex}),
        });
        var d = await r.json();
        if (!r.ok) throw new Error(d.error || 'Камера недоступна');

        status.textContent = '⏳ Прогрев камеры...';
        await new Promise(function(resolve) { setTimeout(resolve, 1500); });

        var ts = Date.now();
        localImg.src = '/mjpeg/raw?' + ts;
        remoteImg.src = '/mjpeg/video?' + ts;

        status.textContent = '✅ ' + (d.info || camLabel) + ' - стрим активен';
        status.className = 'status success';

        statsTimer = setInterval(updateLiveStats, 500);

        btn.textContent = '⏹ Остановить камеру';
        btn.classList.remove('success');
        btn.classList.add('danger');
        btn.disabled = false;
        cameraRunning = true;
        currentMode = 'webcam';

    } catch (e) {
        status.textContent = '❌ ' + e.message;
        status.className = 'status error';
        btn.disabled = false;
        btn.textContent = '▶ Запустить камеру';
    }
}

async function stopCamera() {
    var btn = document.getElementById('camToggleBtn');
    btn.disabled = true;

    await fetch('/mjpeg/stop');

    document.getElementById('localImg').src = '/static/img/placeholder.png';
    document.getElementById('remoteImg').src = '/static/img/placeholder.png';

    if (statsTimer) {
        clearInterval(statsTimer);
        statsTimer = null;
    }

    btn.textContent = '▶ Запустить камеру';
    btn.classList.remove('danger');
    btn.classList.add('success');
    btn.disabled = false;

    document.getElementById('webcamStatus').style.display = 'none';
    cameraRunning = false;
    currentMode = null;
}

//  RTSP тестирование и подключение через WebRTC
document.getElementById('testRtspBtn').addEventListener('click', async function() {
    var url = document.getElementById('rtspUrl').value.trim();
    var status = document.getElementById('rtspStatus');
    if (!url) { showStatus(status, '⚠️️ Введите URL', 'error'); return; }
    showStatus(status, '🔍 Проверка...', 'warning');
    try {
        var r = await fetch('/rtc/test-rtsp', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url: url})
        });
        var d = await r.json();
        showStatus(status, d.ok ? ('✅ ' + d.resolution) : ('❌ ' + d.error),
                   d.ok ? 'success' : 'error');
    } catch (e) {
        showStatus(status, '❌ ' + e.message, 'error');
    }
});

document.getElementById('rtspToggleBtn').addEventListener('click', async function() {
    if (cameraRunning && currentMode === 'rtsp') {
        await stopRtsp();
    } else {
        await startRtsp();
    }
});

document.getElementById('resetBtn').addEventListener('click', async function() {
    await fetch('/rtc/reset');
    document.getElementById('liveTotal').textContent = '0';
    document.getElementById('liveFrame').textContent = '0';
});

function showStatus(el, text, type) {
    el.textContent = text;
    el.className = 'status ' + type;
    el.style.display = 'block';
}

async function updateLiveStats() {
    try {
        var r = await fetch('/rtc/stats');
        var d = await r.json();
        document.getElementById('liveTotal').textContent = d.unique_total || 0;
        document.getElementById('liveFrame').textContent = d.frame_detections || 0;
        document.getElementById('liveFps').textContent = d.fps || 0;
        var tbl = document.getElementById('liveTable');
        var html = '<tr><th>Фрукт</th><th>Количество</th></tr>';
        for (var k in (d.unique_by_class || {})) {
            html += '<tr><td>' + k + '</td><td>' + d.unique_by_class[k] + '</td></tr>';
        }
        tbl.innerHTML = html;
    } catch (e) {}
}

// Функции для работы с RTSP через WebRTC
async function startRtsp() {
    var url = document.getElementById('rtspUrl').value.trim();
    var status = document.getElementById('rtspStatus');
    var btn = document.getElementById('rtspToggleBtn');
    
    if (!url) {
        alert('Введите RTSP URL');
        return;
    }
    
    var remoteImg = document.getElementById('remoteImg');
    var remoteVideo = document.getElementById('remoteVideo');
    var localImg = document.getElementById('localImg');
    var remoteTitle = document.getElementById('remoteTitle');
    
    showStatus(status, '🔌 Подключение к RTSP...', 'warning');
    btn.disabled = true;
    
    try {
        peerConnection = new RTCPeerConnection({
            iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
        });
        
        peerConnection.ontrack = function(event) {
            console.log('[RTSP] ontrack:', event.track.kind);
            if (event.track.kind === 'video') {
                remoteVideo.srcObject = event.streams[0];
                remoteVideo.style.display = 'block';
                remoteImg.style.display = 'none';
                remoteTitle.textContent = '🤖 После AI (RTSP live)';
                
                var playPromise = remoteVideo.play();
                if (playPromise) {
                    playPromise.catch(function(err) {
                        console.warn('play() failed:', err);
                        setTimeout(function() { remoteVideo.play().catch(function(){}); }, 100);
                    });
                }
                
                showStatus(status, '✅ RTSP активен - идёт видео', 'success');
            }
        };
        
        peerConnection.onconnectionstatechange = function() {
            console.log('[RTSP] connectionState:', peerConnection.connectionState);
            if (peerConnection.connectionState === 'failed') {
                showStatus(status, '❌ WebRTC ошибка возможно, брандмауэр блокирует UDP', 'error');
                stopRtsp();
            } else if (peerConnection.connectionState === 'disconnected') {
                showStatus(status, '⚠ Соединение потеряно', 'warning');
            }
        };
        
        var canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, 640, 480);
        
        var dummyStream = canvas.captureStream(1);
        dummyStream.getTracks().forEach(function(t) {
            peerConnection.addTrack(t, dummyStream);
        });
        
        var offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        
        showStatus(status, '📡 Отправка offer на сервер...', 'warning');
        
        var resp = await fetch('/rtc/rtsp-offer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                sdp: peerConnection.localDescription.sdp,
                type: peerConnection.localDescription.type,
                rtsp_url: url,
            })
        });
        
        if (!resp.ok) {
            var errText = await resp.text();
            throw new Error('Сервер вернул ' + resp.status + ': ' + errText);
        }
        
        var answer = await resp.json();
        if (answer.error) throw new Error(answer.error);
        
        await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
        
        localImg.style.display = 'none';
        
        statsTimer = setInterval(updateLiveStats, 500);
        
        btn.textContent = '⏹ Остановить RTSP';
        btn.classList.remove('success');
        btn.classList.add('danger');
        btn.disabled = false;
        
        cameraRunning = true;
        currentMode = 'rtsp';
        
    } catch (e) {
        console.error('[RTSP] Ошибка:', e);
        showStatus(status, '❌ ' + e.message, 'error');
        btn.disabled = false;
    }
}

async function stopRtsp() {
    var btn = document.getElementById('rtspToggleBtn');
    var status = document.getElementById('rtspStatus');
    var remoteImg = document.getElementById('remoteImg');
    var remoteVideo = document.getElementById('remoteVideo');
    var localImg = document.getElementById('localImg');
    var remoteTitle = document.getElementById('remoteTitle');
    
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    
    await fetch('/rtc/rtsp-stop', {method: 'POST'}).catch(function(){});
    
    remoteVideo.srcObject = null;
    remoteVideo.style.display = 'none';
    remoteImg.src = '/static/img/placeholder.png';
    remoteImg.style.display = 'block';
    localImg.src = '/static/img/placeholder.png';
    localImg.style.display = 'block';
    remoteTitle.textContent = '🤖 После AI';
    
    if (statsTimer) {
        clearInterval(statsTimer);
        statsTimer = null;
    }
    
    btn.textContent = '▶ Запустить RTSP';
    btn.classList.remove('danger');
    btn.classList.add('success');
    
    status.style.display = 'none';
    cameraRunning = false;
    currentMode = null;
}

// ИСТОРИЯ
document.getElementById('loadHistoryBtn').addEventListener('click', loadHistory);
document.getElementById('clearHistoryBtn').addEventListener('click', async function() {
    if (!confirm('Очистить историю?')) return;
    await fetch('/history/clear', {method: 'POST'});
    loadHistory();
});
document.getElementById('pdfBtn').addEventListener('click', function() {
    window.location.href = '/report/pdf';
});
document.getElementById('xlsxBtn').addEventListener('click', function() {
    window.location.href = '/report/xlsx';
});

async function loadHistory() {
    try {
        var r = await fetch('/history');
        var data = await r.json();
        var tbl = document.getElementById('historyTable');
        
        var ENGINE_NAMES = {
            'yolo': '⚡ YOLOv8m',
            'owlvit': '🎯 OWLv2',
            'owl-vit': '🎯 OWL-ViT',
        };
        
        var html = '<tr><th>Время</th><th>Файл</th><th>Режим</th>' +
                   '<th>Движок</th><th>Всего</th><th>Детали</th></tr>';
        
        data.forEach(function(h) {
            var bc = h.by_class;
            if (typeof bc === 'string') {
                try { bc = JSON.parse(bc); } catch(e) {}
            }
            
            var detailsStr = '-';
            if (bc && typeof bc === 'object') {
                var parts = [];
                for (var k in bc) {
                    parts.push(k + ':' + bc[k]);
                }
                detailsStr = parts.join(', ') || '-';
            }
            
            var engine = h.engine || 'yolo';
            var engineName = ENGINE_NAMES[engine] || ('❓ ' + engine);
            var modeName = MODE_NAMES[h.mode] || h.mode;
            var time = h.timestamp ? h.timestamp.substring(0, 19).replace('T', ' ') : '-';
            var fname = h.filename || '-';
            if (fname.length > 25) {
                fname = fname.substring(0, 12) + '...' + fname.substring(fname.length - 10);
            }
            
            html += '<tr>';
            html += '<td>' + time + '</td>';
            html += '<td title="' + (h.filename || '') + '">' + fname + '</td>';
            html += '<td>' + modeName + '</td>';
            html += '<td class="engine-cell">' + engineName + '</td>';
            html += '<td><b>' + h.total_fruits + '</b></td>';
            html += '<td class="details-cell" title="' + 
                    (typeof bc === 'object' ? JSON.stringify(bc) : detailsStr) + '">' +
                    detailsStr + '</td>';
            html += '</tr>';
        });
        
        tbl.innerHTML = html;
    } catch (e) {
        console.error('History error:', e);
    }
}

document.getElementById('refreshCamerasBtn').addEventListener('click', loadCameraList);

console.log('[JS] Скрипт загружен успешно');