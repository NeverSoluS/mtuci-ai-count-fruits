# 🍎 AI-подсчёт фруктов на конвейере

Система компьютерного зрения для детекции, сегментации и классификации фруктов в реальном времени с поддержкой веб-камер и RTSP-потоков.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![OWLv2](https://img.shields.io/badge/OWLv2-zero--shot-purple)
![Docker](https://img.shields.io/badge/Docker-ready-blue)



# Оглавление
- [🍎 AI-подсчёт фруктов на конвейере](#-ai-подсчёт-фруктов-на-конвейере)
- [Оглавление](#оглавление)
  - [✨ Возможности](#-возможности)
    - [🎯 Режимы обработки](#-режимы-обработки)
    - [📹 Источники видео](#-источники-видео)
    - [💾 Данные и отчёты](#-данные-и-отчёты)
    - [🐳 Docker](#-docker)
  - [🚀 Быстрый старт (Docker)](#-быстрый-старт-docker)
    - [1. Подготовка](#1-подготовка)
    - [2. Сборка образа](#2-сборка-образа)
    - [3. Запуск контейнера](#3-запуск-контейнера)
    - [4. Просмотр логов](#4-просмотр-логов)
    - [5. Остановка](#5-остановка)
    - [6. Полная очистка (всё удалить)](#6-полная-очистка-всё-удалить)
    - [7. Пересборка после изменений](#7-пересборка-после-изменений)
  - [🐍 Установка без Docker](#-установка-без-docker)
    - [Системные зависимости](#системные-зависимости)
    - [Python-зависимости](#python-зависимости)
    - [Запуск](#запуск)
  - [🎮 Использование](#-использование)
    - [Подключение RTSP-камеры](#подключение-rtsp-камеры)
    - [Проверка RTSP-подключения](#проверка-rtsp-подключения)
  - [⚙️ Конфигурация](#️-конфигурация)
  - [📊 Производительность (CPU)](#-производительность-cpu)
  - [🛠️ Полезные команды Docker](#️-полезные-команды-docker)
  - [🐛 Решение проблем](#-решение-проблем)
    - [RTSP не подключается](#rtsp-не-подключается)
    - [Белый экран при загрузке](#белый-экран-при-загрузке)
    - [Медленная работа OWLv2](#медленная-работа-owlv2)
    - [Контейнер сразу падает](#контейнер-сразу-падает)
  - [📦 Размер образа](#-размер-образа)
  - [📄 Лицензия](#-лицензия)
  - [⚡ Шпаргалка одной командой](#-шпаргалка-одной-командой)


## ✨ Возможности

### 🎯 Режимы обработки

| Режим | Скорость | Фрукты | Описание |
|-------|----------|--------|----------|
| ⚡ **YOLOv8m** | <1 сек | 3 (COCO) | Быстрая детекция bounding boxes |
| 🎯 **OWLv2** | 3-10 сек | 70+ | Zero-shot распознавание любых фруктов |
| 🎭 **Сегментация** | 1-2 сек | 3 | Выделение масок и контуров |
| 🏷️ **Классификация** | <1 сек | Top-5 | Определение типа фрукта |

### 📹 Источники видео
- 📷 **Веб-камеры** - MJPEG-стрим через браузер
- 🎥 **RTSP-потоки** - Hikvision, Dahua, Axis, Reolink, UniFi и др.
- 📁 **Файлы** - загрузка фото и видео

### 💾 Данные и отчёты
- История всех запросов (SQLite)
- Экспорт в **PDF** со сводной статистикой
- Экспорт в **Excel** (2 листа: история + сводка)
- Тёмная/светлая тема с автоопределением

### 🐳 Docker
- CPU-only образ (без NVIDIA, работает на любом железе)
- Docker-outside-of-Docker (управление Docker хоста из контейнера)
- Чистый Docker без docker-compose

---

## 🚀 Быстрый старт (Docker)

### 1. Подготовка

```bash
# Клонировать репозиторий
git clone https://github.com/NeverSoluS/mtuci-ai-count-fruits
cd mtuci-ai-count-fruits

# Создать структуру данных
mkdir -p data/uploads data/results data/reports
chmod -R 777 data/
touch data/history.db
chmod 777 data/history.db

# Установить Docker (если не установлен)
sudo apt install docker.io
```

### 2. Сборка образа

```bash
docker build -t fruit-counter:cpu .
```

### 3. Запуск контейнера

```bash
docker run -d \
  --name fruit-counter \
  --network host \
  -v $(pwd)/data/uploads:/app/static/uploads \
  -v $(pwd)/data/results:/app/static/results \
  -v $(pwd)/data/reports:/app/reports \
  -v $(pwd)/data/history.db:/app/history.db \
  -v fruit-models:/app/.cache \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e TZ=Europe/Moscow \
  --restart unless-stopped \
  fruit-counter:cpu
```

Откройте в браузере: **http://localhost:5000**

### 4. Просмотр логов

```bash
docker logs -f fruit-counter
```

### 5. Остановка

```bash
docker stop fruit-counter
docker rm fruit-counter
```

### 6. Полная очистка (всё удалить)

```bash
# Остановить и удалить контейнер
docker stop fruit-counter
docker rm fruit-counter

# Удалить volume с моделями
docker volume rm fruit-models

# Удалить данные
rm -rf data/

# (Опционально) Удалить образ
docker rmi fruit-counter:cpu
```

### 7. Пересборка после изменений

```bash
docker stop fruit-counter && docker rm fruit-counter
docker build -t fruit-counter:cpu .
# Затем снова запустить через docker run (пункт 3)
```

---

## 🐍 Установка без Docker

### Системные зависимости

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    ffmpeg libavdevice59 libavfilter8 \
    libopus0 libvpx7 libsrtp2-1 libssl3 \
    fonts-dejavu-core
```

**Windows:** Python 3.11 + [FFmpeg](https://ffmpeg.org/download.html) в PATH

**macOS:**
```bash
brew install python@3.11 ffmpeg pkg-config
```

### Python-зависимости

```bash
# Создать виртуальное окружение
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# PyTorch CPU (рекомендуется)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости
pip install -r requirements.txt
```

### Запуск

```bash
python app.py
```

---

## 🎮 Использование

### Подключение RTSP-камеры

Формат URL: `rtsp://USERNAME:PASSWORD@IP:PORT/PATH`

**Примеры популярных камер:**

| Производитель | URL |
|---------------|-----|
| **Hikvision** | `rtsp://admin:pass@IP:554/Streaming/Channels/101` |
| **Dahua** | `rtsp://admin:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| **Axis** | `rtsp://admin:pass@IP:554/axis-media/media.amp` |
| **Reolink** | `rtsp://admin:pass@IP:554/h264Preview_01_main` |
| **UniFi** | `rtsp://IP:7447/cam/realmonitor?channel=1&subtype=0` |
| **Тестовый** | `rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4` |

### Проверка RTSP-подключения

```bash
# Через ffplay
ffplay rtsp://admin:pass@192.168.1.100:554/stream1

# Сохранить кадр
ffmpeg -rtsp_transport tcp -i rtsp://... -frames:v 1 test.jpg
```

---

## ⚙️ Конфигурация

Основные параметры в `config.py`:

```python
# YOLO (быстрая детекция для камеры)
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.11
IMAGE_SIZE = 1920

# OWLv2 (точная детекция для фото)
OWL_CONFIDENCE = 0.31
OWL_MIN_AREA = 300
OWL_NMS_IOU = 0.33

# Модель
OWL_VIT_MODEL = 'google/owlv2-base-patch16-ensemble'
```

---

## 📊 Производительность (CPU)

| Операция | Время |
|----------|-------|
| YOLOv8m (1920×1080) | ~80 мс |
| OWLv2 (1920×1080) | 4-8 сек |
| MJPEG стрим | 25-30 FPS |
| RTSP стрим | 15-25 FPS |

---

## 🛠️ Полезные команды Docker

```bash
# Статус контейнера
docker ps | grep fruit-counter

# Логи в реальном времени
docker logs -f fruit-counter

# Логи за последний час
docker logs --since 1h fruit-counter

# Использование ресурсов
docker stats fruit-counter

# Войти в bash контейнера
docker exec -it fruit-counter bash

# Перезапуск
docker restart fruit-counter

# Размер volume с моделями
docker system df -v | grep fruit-models

# Пересобрать образ без кэша
docker build --no-cache -t fruit-counter:cpu .

# Скопировать файл из контейнера
docker cp fruit-counter:/app/history.db ./backup.db

# Скопировать файл в контейнер
docker cp ./config.py fruit-counter:/app/config.py
```

---

## 🐛 Решение проблем

### RTSP не подключается
- Убедитесь что используется `--network host` в команде `docker run`
- Проверьте доступность: `docker exec fruit-counter ping IP_КАМЕРЫ`
- Посмотрите логи: `docker logs fruit-counter | grep RTSP`

### Белый экран при загрузке
- Очистите кэш браузера: `Ctrl+Shift+R`
- Проверьте консоль браузера (F12) на ошибки


### Медленная работа OWLv2
- Уменьшите `OWL_TOP_K` в `config.py` (например, с 100 до 30)
- Используйте меньшее разрешение изображений

### Контейнер сразу падает
```bash
# Посмотреть последние 50 строк логов
docker logs --tail=50 fruit-counter

# Запустить интерактивно для отладки
docker run --rm -it --network host fruit-counter:cpu bash
```
---

## 📦 Размер образа

| Образ | Размер |
|-------|--------|
| `fruit-counter:cpu` | ~2.5 ГБ |

Модели YOLO (~50 МБ) и OWLv2 (~1.4 ГБ) скачиваются при первом запуске и кэшируются в volume `fruit-models`.


## 📄 Лицензия

MIT License

---

## ⚡ Шпаргалка одной командой

**Запуск с нуля:**
```bash
git clone https://github.com/NeverSoluS/mtuci-ai-count-fruits && \
cd mtuci-ai-count-fruits && \
mkdir -p data/uploads data/results data/reports && \
touch data/history.db && chmod 777 data/history.db && \
docker build -t fruit-counter:cpu . && \
docker run -d --name fruit-counter --network host \
  -v $(pwd)/data/uploads:/app/static/uploads \
  -v $(pwd)/data/results:/app/static/results \
  -v $(pwd)/data/reports:/app/reports \
  -v $(pwd)/data/history.db:/app/history.db \
  -v fruit-models:/app/.cache \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e TZ=Europe/Moscow \
  --restart unless-stopped \
  fruit-counter:cpu
```

**Полная очистка:**
```bash
docker stop fruit-counter && docker rm fruit-counter && \
docker volume rm fruit-models && \
rm -rf data/ && \
docker rmi fruit-counter:cpu
```