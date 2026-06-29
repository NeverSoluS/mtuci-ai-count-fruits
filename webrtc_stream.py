import asyncio
import cv2
import numpy as np
import time
import threading
import os
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from detector import FruitSystem
import config

# Настройка FFmpeg для стабильной работы с RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

class RTSPSource:
    """Улучшенный RTSP source с TCP транспортом и автопереподключением."""
    
    def __init__(self, url):
        self.url = self._prepare_url(url)
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.ready = threading.Event()
        self.reconnect_count = 0
        self.max_reconnects = 10
        threading.Thread(target=self._reader, daemon=True).start()
    
    def _prepare_url(self, url):
        """Подготовка URL с TCP транспортом если не указан."""
        # Если URL уже содержит параметры, добавляем через &
        # Если нет - через ?
        if 'rtsp_transport' not in url:
            separator = '&' if '?' in url else '?'
            url = f"{url}{separator}rtsp_transport=tcp"
        return url
    
    def _connect(self):
        """Подключение к RTSP с настройками стабильности."""
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
            
            print(f"[RTSP] Подключение к {self.url}")
            
            # Создаем VideoCapture с FFmpeg backend
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Настройки для стабильности
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Буфер на 3 кадра
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 сек на подключение
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 сек на чтение
            
            if not self.cap.isOpened():
                print(f"[RTSP] ✗ Не удалось открыть поток")
                return False
            
            print(f"[RTSP] ✓ Подключено успешно")
            return True
            
        except Exception as e:
            print(f"[RTSP] ✗ Ошибка подключения: {e}")
            return False
    
    def _reader(self):
        """Фоновый поток чтения кадров с автопереподключением."""
        consecutive_failures = 0
        max_consecutive_failures = 30  # 30 неудачных чтений = переподключение
        
        while self.running:
            # Подключение если нужно
            if self.cap is None or not self.cap.isOpened():
                if self.reconnect_count >= self.max_reconnects:
                    print(f"[RTSP] ✗ Превышено количество переподключений ({self.max_reconnects})")
                    time.sleep(5)
                    self.reconnect_count = 0
                
                if not self._connect():
                    self.reconnect_count += 1
                    time.sleep(2)
                    continue
                
                consecutive_failures = 0
            
            # Чтение кадра
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Успешное чтение
                    with self.lock:
                        self.frame = frame.copy()
                    self.ready.set()
                    consecutive_failures = 0
                else:
                    # Неудачное чтение
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"[RTSP] ⚠ Потеряно {consecutive_failures} кадров подряд, переподключение...")
                        self.reconnect_count += 1
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None
                        time.sleep(1)
                    else:
                        time.sleep(0.05)  # Короткая пауза перед следующей попыткой
                        
            except Exception as e:
                print(f"[RTSP] ✗ Ошибка чтения: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
    
    def get_frame(self):
        """Получить последний кадр (потокобезопасно)."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def wait_ready(self, timeout=10.0):
        """Ожидание первого кадра."""
        return self.ready.wait(timeout)
    
    def stop(self):
        """Остановка потока."""
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass


class VideoTransformTrack(MediaStreamTrack):
    """Трек для веб-камеры."""
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.detector = FruitSystem(mode='detection')
        self.frame_count = 0
        self.start_time = time.time()

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        annotated, stats = self.detector.process_frame(img, tracking=True, fast=True)

        self.frame_count += 1
        fps = self.frame_count / (time.time() - self.start_time + 1e-6)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        latest_stats.update({
            'unique_total': stats.get('total', 0),
            'unique_by_class': stats.get('by_class', {}),
            'frame_detections': stats.get('frame_detections', 0),
            'fps': round(fps, 1),
        })

        new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


class RTSPTransformTrack(MediaStreamTrack):
    """Трек для RTSP потока."""
    kind = "video"

    def __init__(self, url):
        super().__init__()
        self.source = RTSPSource(url)
        self.detector = FruitSystem(mode='detection')
        self.frame_count = 0
        self.start_time = time.time()
        self.pts = 0
        
        print(f"[RTSP] Ожидание первого кадра...")
        if self.source.wait_ready(timeout=10.0):
            print(f"[RTSP] ✓ Первый кадр получен")
        else:
            print(f"[RTSP] ⚠ Таймаут ожидания первого кадра")

    async def recv(self):
        frame = self.source.get_frame()
        
        if frame is None:
            # Заглушка если нет сигнала
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No RTSP signal", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Обработка через AI
            try:
                annotated, stats = self.detector.process_frame(frame, tracking=True, fast=True)
                
                self.frame_count += 1
                fps = self.frame_count / (time.time() - self.start_time + 1e-6)
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                latest_stats.update({
                    'unique_total': stats.get('total', 0),
                    'unique_by_class': stats.get('by_class', {}),
                    'frame_detections': stats.get('frame_detections', 0),
                    'fps': round(fps, 1),
                })
                
                frame = annotated
            except Exception as e:
                print(f"[RTSP] ✗ Ошибка обработки: {e}")
        
        # Создаем VideoFrame
        self.pts += 3000
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = self.pts
        new_frame.time_base = "1/90000"
        return new_frame

    def stop(self):
        """Остановка трека."""
        self.source.stop()


# Глобальная статистика
latest_stats = {
    'unique_total': 0,
    'unique_by_class': {},
    'frame_detections': 0,
    'fps': 0,
}


async def offer(request_json):
    """WebRTC offer для веб-камеры."""
    pc = RTCPeerConnection()

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[WebRTC] state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()

    @pc.on("track")
    def on_track(track):
        print(f"[WebRTC] track: {track.kind}")
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(track))

    offer_desc = RTCSessionDescription(sdp=request_json["sdp"], type=request_json["type"])
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def rtsp_offer(request_json):
    """WebRTC offer для RTSP потока."""
    pc = RTCPeerConnection()
    url = request_json.get("rtsp_url", "")
    
    print(f"[RTSP] Создание трека для {url}")

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[RTSP] state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()

    # Добавляем RTSP трек
    rtsp_track = RTSPTransformTrack(url)
    pc.addTrack(rtsp_track)

    offer_desc = RTCSessionDescription(sdp=request_json["sdp"], type=request_json["type"])
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    print(f"[RTSP] ✓ Offer готов")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def get_stats():
    """Получить статистику."""
    return latest_stats


def reset_detector():
    """Сбросить счетчики."""
    latest_stats.update({
        'unique_total': 0,
        'unique_by_class': {},
        'frame_detections': 0,
    })