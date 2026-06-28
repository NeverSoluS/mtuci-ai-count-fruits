from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from werkzeug.utils import secure_filename
import os, time, asyncio, cv2, threading
import numpy as np

import config
from detector import FruitSystem
from database import init_db, save_request, get_all_requests, clear_history
from reports import generate_pdf, generate_excel
from webrtc_stream import offer, rtsp_offer, get_stats, reset_detector

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.RESULT_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)
init_db()

print("\n[INIT] Загрузка моделей...")
detection_system = FruitSystem(mode='detection')
segmentation_system = FruitSystem(mode='segmentation')
classification_system = FruitSystem(mode='classification')


# Цель: один захват с камеры = много клиентов MJPEG
class CameraStreamer:
    """Один захват = много MJPEG-клиентов."""

    def __init__(self):
        # Инициализация
        self.cap = None
        self.detector = FruitSystem(mode='detection')
        self.running = False
        self.clients = 0
        self.lock = threading.Lock()
        # Буферы кадров
        self.raw_frame = None
        self.ai_frame = None
        self.frame_id = 0
        self.frame_event = threading.Event()
        # Статистика
        self.frame_count = 0
        self.start_time = None
        self.camera_info = ""
        self.warmup_frames = 0
        self._thread = None

    # Запуск камеры по индексу
    def start(self, camera_index=0):
        """Запустить камеру по индексу (из enumerateDevices)."""
        with self.lock:
            if self.running:
                self.clients += 1
                print(f"[Camera] Уже запущена, clients={self.clients}")
                return True

            # Пробуем открыть камеру по индексу
            self.cap = self._open_camera(camera_index)
            if not self.cap or not self.cap.isOpened():
                return False

            # Настройки
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Инициализация статистики
            self.running = True
            self.clients = 1
            self.start_time = time.time()
            self.frame_count = 0
            self.frame_id = 0
            self.warmup_frames = 5
            self.frame_event.clear()
            self.camera_info = f"index={camera_index}"
            # Запуск потока чтения кадров
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            print(f"[Camera] Запущена: {self.camera_info}")
            return True

    # Открытие камеры с разными бэкендами
    def _open_camera(self, index):
        """Открыть камеру по индексу с разными бэкендами."""
        backends = [
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_ANY, "ANY"),
        ]
        # Пробуем открыть камеру с разными бэкендами
        for backend, name in backends:
            try:
                print(f"[Camera] Пробую index={index}, backend={name}")
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    # Проверяем чтение кадра
                    for _ in range(3):
                        ret, frame = cap.read()
                        if ret and self._is_valid_frame(frame):
                            self.camera_info = f"index={index}, backend={name}, {frame.shape[1]}x{frame.shape[0]}"
                            self.raw_frame = frame.copy()
                            self.ai_frame = frame.copy()
                            return cap
                    cap.release()
            except Exception as e:
                print(f"[Camera] Ошибка {name}: {e}")
                continue
        return None

    # Проверка валидности кадра
    def _is_valid_frame(self, frame):
        if frame is None or frame.size == 0:
            return False
        mean_val = np.mean(frame)
        std_val = np.std(frame)
        return 5 < mean_val < 250 and std_val > 2
    # Остановка клиента
    def stop_client(self):
        with self.lock:
            if self.clients <= 0:
                return
            self.clients -= 1
            print(f"[Camera] Клиент отключён, clients={self.clients}")
            if self.clients <= 0:
                self.running = False
                self.frame_event.set()  # разбудить ждущих
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.raw_frame = None
                self.ai_frame = None
                self.camera_info = ""
                print("[Camera] Остановлена")

    # Поток чтения кадров
    def _reader_loop(self):
        """Читает кадры, обновляет буферы, будит клиентов."""
        print("[Camera] reader_loop запущен")
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            # Чтение кадра
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            # Пропускаем первые кадры для стабилизации
            if self.warmup_frames > 0:
                self.warmup_frames -= 1
                continue
                # Проверка валидности кадра
            if not self._is_valid_frame(frame):
                continue
            
            raw = frame.copy()
            self.raw_frame = raw
            # AI-обработка кадра
            try:
                annotated, stats = self.detector.process_frame(frame, tracking=True, fast=True)
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / (elapsed + 1e-6)
                cv2.putText(annotated, f"FPS: {fps:.1f}",
                            (10, annotated.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                # Каждые 30 кадров выводим статистику в консоль
                if self.frame_count % 30 == 0:
                    det = stats.get('frame_detections', 0)
                    print(f"[Camera] #{self.frame_count}, FPS: {fps:.1f}, "
                          f"Обнаружено: {det} | {self.camera_info}")

                # Обновляем глобальную статистику для /rtc/stats
                from webrtc_stream import latest_stats
                latest_stats['unique_total'] = stats.get('total', 0)
                latest_stats['unique_by_class'] = stats.get('by_class', {})
                latest_stats['frame_detections'] = stats.get('frame_detections', 0)
                latest_stats['fps'] = round(fps, 1)
                # Обновляем буфер AI-кадра
                self.ai_frame = annotated
            except Exception as e:
                print(f"[Camera] Ошибка AI: {e}")
                self.ai_frame = raw.copy()

            # Увеличиваем ID и будим всех клиентов
            self.frame_id += 1
            self.frame_event.set()
            # Даём время клиентам забрать кадр
            time.sleep(0.001)
            self.frame_event.clear()
        print("[Camera] reader_loop завершился")

    # Ждать пока frame_id изменится
    def wait_for_new_frame(self, current_id, timeout=1.0):
        deadline = time.time() + timeout
        while time.time() < deadline and self.running:
            if self.frame_id != current_id:
                return True
            self.frame_event.wait(timeout=0.05)
        return False

# Инициализация глобального стримера
camera_streamer = CameraStreamer()

# Функция кодирования кадра в JPEG
def _encode_jpeg(frame, quality=85):
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Нет сигнала", (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    ret, jpeg = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, quality,
                               cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    return jpeg.tobytes() if ret else b''

# MJPEG-генератор
def mjpeg_stream_generator(get_frame, quality=85):
    """
    Бесконечный MJPEG-генератор.
    get_frame: функция = (frame, frame_id)
    """
    last_id = -1
    while camera_streamer.running:
        frame, current_id = get_frame()

        # Если кадр новый - отправляем
        if current_id != last_id:
            jpeg_bytes = _encode_jpeg(frame, quality=quality)
            if jpeg_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Cache-Control: no-cache, no-store\r\n\r\n' +
                       jpeg_bytes + b'\r\n')
                last_id = current_id
        else:
            # Ждём новый кадр
            if not camera_streamer.wait_for_new_frame(current_id, timeout=1.0):
                # Таймаут - отправим текущий кадр повторно
                jpeg_bytes = _encode_jpeg(frame, quality=quality)
                if jpeg_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           jpeg_bytes + b'\r\n')

# Вспомогательная функция для генерации одного кадра с текстом
def _single_frame(text):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    cv2.putText(frame, text, (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    jpeg = _encode_jpeg(frame, quality=70)
    return (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


# Маршруты Flask
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для favicon.ico
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'favicon.ico',mimetype='image/vnd.microsoft.icon')

# Маршрут для обработки загруженного файла
@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify(error='Файл не загружен'), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify(error='Пустое имя'), 400
        # Получаем параметры
        mode = request.form.get('mode', 'detection')
        engine = request.form.get('engine', 'yolo')
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}:
            return jsonify(error=f'Формат не поддерживается: {ext}'), 400
        # Сохраняем файл
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        input_path = os.path.join(config.UPLOAD_DIR, filename)
        file.save(input_path)
        # Определяем, видео это или изображение
        is_video = ext in {'mp4', 'avi', 'mov'}

        # Выбор системы
        if is_video or mode != 'detection' or engine != 'owlvit':
            systems = {'detection': detection_system, 'segmentation': segmentation_system,
                       'classification': classification_system}
            system = systems.get(mode, detection_system)
        else:
            from detector import get_owlvit_detector
            try:
                system = get_owlvit_detector()
            except Exception as e:
                print(f"[OWL-ViT] Ошибка, fallback на YOLO: {e}")
                system = detection_system
        # Сбрасываем трекер, чтобы не было старых объектов
        system.reset_tracker()
        # Проверка: видео только в детекции
        if is_video and mode != 'detection':
            return jsonify(error='Видео только в детекции'), 400
        # Генерация имени выходного файла   
        output_filename = filename if not is_video else filename.rsplit('.', 1)[0] + '.mp4'
        output_path = os.path.join(config.RESULT_DIR, output_filename)

        t0 = time.time()
        
        # Определение названия движка
        engine_name = 'YOLOv8m'
        if hasattr(system, 'is_owlv2'):
            engine_name = 'OWLv2' if system.is_owlv2 else 'OWL-ViT v1'
        elif hasattr(system, 'model_name'):
            engine_name = system.model_name.split('/')[-1]
        
        print(f"\n[PROCESS] Режим: {mode}, Движок: {engine_name}")
        print(f"[PROCESS] Файл: {filename}")
        
        # вызываем обработку
        stats = (system.process_video if is_video else system.process_image)(input_path, output_path)
        stats['processing_time_sec'] = round(time.time() - t0, 2)
        stats['mode'] = mode
        
        # Сохраняем engine
        if 'engine' not in stats:
            stats['engine'] = 'owlvit' if 'owl' in engine_name.lower() else 'yolo'
        
        print(f"[PROCESS] Результат: {stats.get('total', 0)} фруктов за {stats['processing_time_sec']}с")
        print(f"[PROCESS] Движок: {stats['engine']}")
        # Сохраняем в базу
        try:
            save_request(filename, stats, output_filename)
        except Exception:
            pass
        # Возвращаем JSON с результатами
        return jsonify({
            'count': stats.get('total', 0),
            'by_class': stats.get('by_class', {}),
            'top5': stats.get('top5', {}),
            'processing_time': stats['processing_time_sec'],
            'result_url': output_path,
            'is_video': is_video,
            'mode': mode,
            'engine': stats.get('engine', 'yolo'),
        })
    except Exception as e:
        import traceback
        return jsonify(error=f"{e}\n{traceback.format_exc()}"), 500

# Маршруты для MJPEG и WebRTC
@app.route('/mjpeg/start', methods=['POST'])
def mjpeg_start():
    data = request.get_json() or {}
    index = int(data.get('index', 0))

    # Если уже запущена с другим индексом - перезапускаем
    if camera_streamer.running:
        current_idx = int(camera_streamer.camera_info.split('=')[1].split(',')[0]) \
            if camera_streamer.camera_info.startswith('index=') else -1
        if current_idx != index:
            camera_streamer.stop_client()
            time.sleep(0.3)
    # Запуск камеры
    if camera_streamer.start(camera_index=index):
        return jsonify(status='ok', info=camera_streamer.camera_info)
    return jsonify(error='Не удалось открыть камеру'), 500

# Маршрут для остановки MJPEG
@app.route('/mjpeg/stop')
def mjpeg_stop():
    camera_streamer.stop_client()
    return jsonify(status='ok')

# Маршрут для MJPEG raw
@app.route('/mjpeg/raw')
def mjpeg_raw():
    """Оригинальное видео."""
    if not camera_streamer.running:
        return Response(_single_frame("Камера не запущена"),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    # Генератор кадров
    def get():
        return (camera_streamer.raw_frame, camera_streamer.frame_id)
    # Возвращаем поток MJPEG
    return Response(mjpeg_stream_generator(get, quality=80),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Маршрут для MJPEG с AI-обработкой
@app.route('/mjpeg/video')
def mjpeg_video():
    """Видео с AI-обработкой."""
    if not camera_streamer.running:
        return Response(_single_frame("Камера не запущена"),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    # Генератор кадров
    def get():
        return (camera_streamer.ai_frame, camera_streamer.frame_id)
    # Возвращаем поток MJPEG
    return Response(mjpeg_stream_generator(get, quality=85),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Маршрут для WebRTC offer
@app.route('/rtc/offer', methods=['POST'])
def rtc_offer():
    data = request.get_json()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        answer = loop.run_until_complete(offer(data))
    finally:
        loop.close()
    return jsonify(answer)

# Маршрут для WebRTC RTSP offer
@app.route('/rtc/rtsp-offer', methods=['POST'])
def rtc_rtsp_offer():
    data = request.get_json()
    if not data.get('rtsp_url', '').startswith('rtsp://'):
        return jsonify(error='Не RTSP URL'), 400
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        answer = loop.run_until_complete(rtsp_offer(data))
    finally:
        loop.close()
    return jsonify(answer)

# Маршрут для теста RTSP
@app.route('/rtc/test-rtsp', methods=['POST'])
def test_rtsp():
    url = request.json.get('url', '')
    if not url:
        return jsonify(ok=False, error='URL пустой'), 400
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return jsonify(ok=False, error='Не удалось подключиться')
    ret, frame = cap.read()
    cap.release()
    return jsonify(ok=ret,
                   resolution=f"{frame.shape[1]}x{frame.shape[0]}" if ret else None)

# Маршрут для остановки RTSP стрима
@app.route('/rtc/rtsp-stop', methods=['POST'])
def rtc_rtsp_stop():
    """Остановить активный RTSP стрим."""
    # Сбрасываем статистику
    reset_detector()
    return jsonify(status='ok')

# Маршрут для получения статистики RTSP
@app.route('/rtc/stats')
def rtc_stats():
    return jsonify(get_stats())

# Маршрут для сброса детектора
@app.route('/rtc/reset')
def rtc_reset():
    reset_detector()
    return jsonify(status='ok')

# Маршрут для истории запросов
@app.route('/history')
def history():
    return jsonify(get_all_requests())

# Маршрут для очистки истории
@app.route('/history/clear', methods=['POST'])
def history_clear():
    clear_history()
    return jsonify(status='ok')

# Маршрут для генерации отчета
@app.route('/report/<fmt>')
def report(fmt):
    data = get_all_requests()
    path = generate_pdf(data) if fmt == 'pdf' else generate_excel(data) if fmt == 'xlsx' else None
    if not path:
        return jsonify(error='Формат не поддерживается'), 400
    return send_file(path, as_attachment=True)

# Главный запуск Flask
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🍎 AI-система подсчёта фруктов")
    print(f"Сервер: http://localhost:{config.PORT}")
    print(f"{'='*60}\n")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)