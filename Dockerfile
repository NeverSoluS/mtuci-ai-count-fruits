# AI-подсчёт фруктов - Production Docker image
# Multi-stage build для минимального размера


# Stage 1: Builder - установка зависимостей и сборка виртуального окружения
FROM python:3.11-slim as builder

WORKDIR /build

# Системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости в виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# PyTorch CPU-версия (меньше размер, без CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Final image - минимальный образ с приложением
FROM python:3.11-slim

# Метаданные
LABEL maintainer="fruit-counter"
LABEL description="AI-система подсчёта фруктов с YOLOv8 и OWLv2"

# Рабочая директория
WORKDIR /app

# Системные библиотеки для OpenCV, aiortc, ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # FFmpeg для RTSP/видео
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    libssl-dev \
    # Шрифты с кириллицей
    fonts-dejavu-core \
    fonts-liberation \
    # Утилиты
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем виртуальное окружение из builder-стейджа
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Переменные окружения Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Кэш моделей HuggingFace (для OWLv2)
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Кэш моделей Ultralytics (для YOLO)
    YOLO_CACHE_DIR=/app/.cache/ultralytics \
    # OpenCV: используем TCP для RTSP по умолчанию
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|stimeout;5000000"

# Копируем исходный код
COPY . .

# Создаём директории для данных
RUN mkdir -p \
    static/uploads \
    static/results \
    static/img \
    reports \
    .cache/huggingface \
    .cache/ultralytics \
    && chmod -R 777 static reports .cache

# Предзагрузка YOLO-моделей (опционально, ускоряет первый запуск)
# Раскомментируйте если хотите модели внутри образа (+~100 MB)
# RUN python -c "from ultralytics import YOLO; \
#     YOLO('yolov8m.pt'); \
#     YOLO('yolov8m-seg.pt'); \
#     YOLO('yolov8n-cls.pt')"

# Healthcheck - проверка доступности сервера
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Открываем порт Flask
EXPOSE 5000

# Переключаемся на непривилегированного пользователя
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Запуск приложения
CMD ["python", "app.py"]