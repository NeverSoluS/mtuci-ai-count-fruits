"""
Детектор фруктов с четырьмя режимами:
1. Детекция (YOLOv8)  - быстро, 3 фрукта COCO
2. Сегментация (YOLOv8-seg)  - маски
3. Классификация (YOLOv8-cls)  - Top-5
4. OWLv2 (zero-shot)  - все 70+ фруктов, медленно
"""

import cv2
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
import functools
import traceback

from ultralytics import YOLO
import config



# Локальные переменные для OWL-ViT/OWLv2
OWLVIT_AVAILABLE = False
torch = None
torchvision_nms = None
OwlViTProcessor = None
OwlViTForObjectDetection = None

# Загрузка OWL-ViT/OWLv2 по требованию (lazy load) для экономии памяти и времени старта
def _load_owlvit():
    """Загрузить библиотеки OWL-ViT/OWLv2 по требованию."""
    global OWLVIT_AVAILABLE, torch, torchvision_nms
    global OwlViTProcessor, OwlViTForObjectDetection
    if OWLVIT_AVAILABLE:
        return True
    try:
        import torch as _torch
        # ИСПРАВЛЕНО: используем Auto-классы, которые сами выберут правильную архитектуру
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        torch = _torch
        OwlViTProcessor = AutoProcessor           # сам выберет OwlViTProcessor или Owlv2Processor
        OwlViTForObjectDetection = AutoModelForZeroShotObjectDetection  # сам выберет v1 или v2
        try:
            from torchvision.ops import nms as _nms
            torchvision_nms = _nms
        except ImportError:
            torchvision_nms = None
        OWLVIT_AVAILABLE = True
        print("[OWL] Библиотеки загружены (AutoProcessor + AutoModel)")
        return True
    except ImportError as e:
        print(f"[OWL] Не установлен: {e}")
        print("  Установите: pip install transformers torch torchvision")
        return False


# Функции для отрисовки текста с кириллицей через PIL (с кэшированием шрифтов)
@functools.lru_cache(maxsize=16)
def _get_font(size):
    """Получить шрифт с поддержкой кириллицы (с кэшированием)."""
    for path in ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/Arial.ttf',
                 'arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/System/Library/Fonts/Helvetica.ttc']:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

# Функция для отрисовки текста с кириллицей через PIL
def draw_text(img, text, pos, size=20, color=(0, 255, 0), bg=None, fast=False):
    """Отрисовка текста с кириллицей через PIL."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = _get_font(size)
    x, y = pos

    if bg and not fast:
        try:
            bbox = draw.textbbox((x, y), text, font=font)
            bbox = (bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2)
            draw.rectangle(bbox, fill=(bg[2], bg[1], bg[0]))
        except Exception:
            pass

    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# Функция для стабильного цвета по хешу имени класса
def _hash_color(label):
    """Стабильный цвет для каждого класса (по хешу имени)."""
    import hashlib
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF
    return (b, g, r)  # BGR для OpenCV

# Класс для обработки изображений и видео с разными режимами (детекция, сегментация, классификация, OWLv2)
class FruitSystem:
    """Система для трёх типов задач на базе YOLO."""

    def __init__(self, mode='detection'):
        self.mode = mode
        self._load_model()
        self.unique_tracks = set()
        self.track_counts = defaultdict(int)
        self._track_id_counter = 0
        self._last_centroids = {}

    def _load_model(self):
        if self.mode == 'detection':
            print(f"[INFO] Загрузка детекции: {config.DETECTION_MODEL}")
            self.model = YOLO(config.DETECTION_MODEL)
        elif self.mode == 'segmentation':
            print(f"[INFO] Загрузка сегментации: {config.SEGMENTATION_MODEL}")
            self.model = YOLO(config.SEGMENTATION_MODEL)
        elif self.mode == 'classification':
            print(f"[INFO] Загрузка классификации: {config.CLASSIFICATION_MODEL}")
            self.model = YOLO(config.CLASSIFICATION_MODEL)

    def reset_tracker(self):
        self.unique_tracks.clear()
        self.track_counts.clear()
        self._track_id_counter = 0
        self._last_centroids.clear()

    def _simple_track(self, detections):
        matched = {}
        for (x1, y1, x2, y2, label) in detections:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            best_id, best_dist = None, 1e9
            for tid, (ocx, ocy, ol) in self._last_centroids.items():
                if ol != label:
                    continue
                dist = ((cx-ocx)**2 + (cy-ocy)**2)**0.5
                if dist < best_dist and dist < max(x2-x1, y2-y1)*0.5:
                    best_id, best_dist = tid, dist
            if best_id is not None:
                matched[best_id] = (cx, cy, label)
            else:
                self._track_id_counter += 1
                matched[self._track_id_counter] = (cx, cy, label)
                self.unique_tracks.add(self._track_id_counter)
                self.track_counts[label] += 1
        self._last_centroids = matched

    def process_frame(self, frame, tracking=True, fast=False):
        if self.mode == 'detection':
            return self._detect(frame, tracking, fast)
        elif self.mode == 'segmentation':
            return self._segment(frame, fast)
        else:
            return self._classify(frame, fast)

    def _detect(self, frame, tracking, fast):
        results = self.model(frame,
                             conf=config.CONFIDENCE_THRESHOLD,
                             iou=config.IOU_THRESHOLD,
                             imgsz=config.IMAGE_SIZE,
                             max_det=config.MAX_DETECTIONS,
                             verbose=False)
        res = results[0]
        counts = defaultdict(int)
        annotated = frame.copy()
        detections = []

        if res.boxes is not None:
            for box in res.boxes:
                cls_name = res.names[int(box.cls[0])]
                if cls_name not in config.FRUIT_CLASSES_COCO:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ru = config.translate_label(cls_name)
                detections.append((x1, y1, x2, y2, ru))
                counts[ru] += 1

                color = _hash_color(ru)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                annotated = draw_text(annotated, f"{ru} {conf:.2f}",
                                      (x1, y1-25), 18, (255, 255, 255), color, fast)

        if tracking and detections:
            self._simple_track(detections)

        total = len(detections)
        annotated = draw_text(annotated, f"Фруктов: {total}", (10, 10), 28,
                              (255, 255, 255), (0, 100, 0), fast)

        return annotated, {
            'total': len(self.unique_tracks) if tracking else total,
            'by_class': dict(self.track_counts) if tracking else dict(counts),
            'frame_detections': total,
            'mode': 'detection',
            'engine': 'yolo',
        }

    def _segment(self, frame, fast):
        results = self.model(frame,
                             conf=config.CONFIDENCE_THRESHOLD,
                             iou=config.IOU_THRESHOLD,
                             verbose=False)
        res = results[0]
        counts = defaultdict(int)
        annotated = frame.copy()

        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()
            boxes = res.boxes
            h, w = frame.shape[:2]
            for i, mask in enumerate(masks):
                cls_name = res.names[int(boxes.cls[i])]
                if cls_name not in config.FRUIT_CLASSES_COCO:
                    continue
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                ru = config.translate_label(cls_name)
                counts[ru] += 1

                mask_r = cv2.resize(mask, (w, h))
                bin_mask = (mask_r > 0.5).astype(np.uint8)
                color = _hash_color(ru)
                overlay = annotated.copy()
                overlay[bin_mask == 1] = color
                annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, color, 2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                annotated = draw_text(annotated, f"{ru} {conf:.2f}",
                                      (x1, y1-25), 18, (255, 255, 255), color, fast)

        total = sum(counts.values())
        annotated = draw_text(annotated, f"Фруктов: {total}", (10, 10), 30,
                              (255, 255, 255), (0, 0, 150), fast)
        return annotated, {
            'total': total,
            'by_class': dict(counts),
            'frame_detections': total,
            'mode': 'segmentation',
            'engine': 'yolo',
        }

    def _classify(self, frame, fast):
        results = self.model(frame, verbose=False)
        res = results[0]
        annotated = frame.copy()

        top5 = res.probs.top5
        top5conf = res.probs.top5conf.cpu().numpy()
        names = res.names

        top1_name = config.translate_label(names[res.probs.top1])
        top1_conf = float(top5conf[0])
        annotated = draw_text(annotated, f"Класс: {top1_name} ({top1_conf:.0%})",
                              (10, 10), 32, (255, 255, 255), (150, 0, 0), fast)

        y = 60
        top5_dict = {}
        for idx, conf in zip(top5, top5conf):
            ru = config.translate_label(names[idx])
            annotated = draw_text(annotated, f"{ru}: {conf:.0%}",
                                  (10, y), 24, (0, 255, 0), (0, 100, 0), fast)
            top5_dict[ru] = float(conf)
            y += 35

        return annotated, {
            'total': 1,
            'by_class': {top1_name: 1.0},
            'top5': top5_dict,
            'frame_detections': 1,
            'mode': 'classification',
            'engine': 'yolo',
        }

    def process_image(self, image_path, output_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Не удалось прочитать: {image_path}")
        annotated, stats = self.process_frame(frame, tracking=False, fast=False)
        cv2.imwrite(output_path, annotated)
        return stats

    def process_video(self, video_path, output_path):
        if self.mode != 'detection':
            raise ValueError("Видео только в режиме детекции")

        self.reset_tracker()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (int(cap.get(3)), int(cap.get(4))))

        frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, _ = self.process_frame(frame, tracking=True, fast=True)
            out.write(annotated)
            frames += 1

        cap.release()
        out.release()
        return {
            'total': len(self.unique_tracks),
            'by_class': dict(self.track_counts),
            'frame_detections': frames,
            'mode': 'detection',
            'engine': 'yolo',
        }


# OWL-ViT / OWLv2 детектор (zero-shot)
_owlvit_instance = None


def get_owlvit_detector():
    global _owlvit_instance
    if _owlvit_instance is None:
        _owlvit_instance = OwlVitDetector()
    return _owlvit_instance

# Класс для обработки изображений с OWL-ViT / OWLv2 (zero-shot)
class OwlVitDetector:
    def __init__(self):
        if not _load_owlvit():
            raise RuntimeError("OWL-ViT не установлен")

        model_name = config.OWL_VIT_MODEL
        try:
            print(f"[OWL] Загрузка: {model_name}")
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        except Exception as e:
            print(f"[OWL] Fallback на owlvit-base-patch32: {e}")
            model_name = 'google/owlvit-base-patch32'
            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name)

        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.use_fp16 = False
        if self.device == 'cuda':
            try:
                self.model = self.model.half()
                self.use_fp16 = True
                print(f"[OWL] ✓ FP16 на GPU")
            except Exception:
                pass

        self.model_name = model_name
        self.is_owlv2 = 'owlv2' in model_name.lower()

        print(f"[OWL] Загружен: {model_name} на {self.device}")
        print(f"[OWL] Режим: {'OWLv2' if self.is_owlv2 else 'OWL-ViT v1'}")
        print(f"[OWL] Классов: {len(config.FRUIT_LABELS_EN)}")
        
        # Определяем какой post_process метод доступен
        self.postprocess_fn = None
        self.postprocess_name = None
        for name in ['post_process_object_detection', 
                     'post_process_grounded_object_detection']:
            if hasattr(self.processor, name):
                self.postprocess_fn = getattr(self.processor, name)
                self.postprocess_name = name
                break
        
        if self.postprocess_fn is None:
            raise RuntimeError("У процессора нет post_process метода!")
        
        print(f"[OWL] Используем: {self.postprocess_name}")

    def reset_tracker(self):
        pass

    def process_image(self, image_path, output_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Не удалось прочитать: {image_path}")
        try:
            annotated, stats = self._process_frame(frame)
            cv2.imwrite(output_path, annotated)
            return stats
        except Exception as e:
            print(f"[OWL] ОШИБКА: {e}")
            traceback.print_exc()
            raise

    def process_video(self, video_path, output_path):
        raise ValueError("OWL-ViT не поддерживает видео")

    def process_frame(self, frame, tracking=True, fast=False):
        return self._process_frame(frame)

    def _process_frame(self, frame):
        """Основная логика с правильной постобработкой."""
        print(f"\n[OWL] ====== Начало ======")
        print(f"[OWL] Кадр: {frame.shape}")

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Компоновка текстов для zero-shot
        texts = [config.FRUIT_LABELS_EN]

        inputs = self.processor(
            text=texts,
            images=pil_image,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)

        if self.use_fp16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        print(f"[OWL] Инференс...")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Размеры оригинального изображения для постобработки
        target_sizes = torch.Tensor([[h, w]]).to(self.device)

        # Постобработка через post_process_object_detection или post_process_grounded_object_detection
        print(f"[OWL] Постобработка через {self.postprocess_name}...")
        
        try:
            # Передаём input_ids только если метод их принимает
            import inspect
            sig = inspect.signature(self.postprocess_fn)
            accepts_input_ids = 'input_ids' in sig.parameters
            accepts_nms = 'nms_threshold' in sig.parameters
            
            kwargs = {
                'threshold': config.OWL_CONFIDENCE,
                'target_sizes': target_sizes,
            }
            
            if self.postprocess_name == 'post_process_grounded_object_detection' and accepts_input_ids:
                kwargs['input_ids'] = inputs.input_ids
            
            if accepts_nms:
                kwargs['nms_threshold'] = config.OWL_NMS_IOU
            
            print(f"[OWL] post_process kwargs: {list(kwargs.keys())}")
            results = self.postprocess_fn(outputs, **kwargs)
            
        except Exception as e:
            print(f"[OWL] Ошибка post_process: {e}")
            # Fallback
            try:
                results = self.processor.post_process_object_detection(
                    outputs,
                    threshold=config.OWL_CONFIDENCE,
                    target_sizes=target_sizes,
                )
            except Exception as e2:
                print(f"[OWL] Повторная ошибка: {e2}")
                raise

        # results может быть списком (для batch) или словарём (для одного изображения)
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        print(f"[OWL] result keys: {list(result.keys())}")

        # Извлекаем результаты
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        # Переводим в numpy
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

        print(f"[OWL] Найдено {len(boxes)} объектов до фильтрации")
        if len(scores) > 0:
            print(f"[OWL] scores range: [{scores.min():.3f}, {scores.max():.3f}]")
            # Покажем первые 5 для отладки
            for i in range(min(5, len(scores))):
                idx = int(labels[i])
                en = config.FRUIT_LABELS_EN[idx] if idx < len(config.FRUIT_LABELS_EN) else f"?{idx}"
                ru = config.translate_label(en)
                print(f"[OWL]   #{i+1}: {ru} ({scores[i]:.3f})")

        # Фильтрация по площади и сбор детекций
        detections = []
        min_area = getattr(config, 'OWL_MIN_AREA', 400)
        max_aspect = getattr(config, 'OWL_ASPECT_RATIO_MAX', 3.0)

        for i in range(len(scores)):
            idx = int(labels[i])
            if idx >= len(config.FRUIT_LABELS_EN):
                continue

            en = config.FRUIT_LABELS_EN[idx]
            ru = config.translate_label(en)
            conf = float(scores[i])

            # Координаты уже в пикселях (xyxy) после post_process
            x1, y1, x2, y2 = map(int, boxes[i])

            # Проверка валидности
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            if x2 <= x1 or y2 <= y1:
                continue

            # Фильтр по минимальной площади
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                continue

            # Фильтр aspect ratio
            aspect = max(x2-x1, y2-y1) / (min(x2-x1, y2-y1) + 1)
            if aspect > max_aspect:
                continue

            # Обрезка по границам
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            detections.append((x1, y1, x2, y2, ru, conf))

        print(f"[OWL] После фильтров: {len(detections)}")

        # Дополнительный NMS (на случай если post_process его не делал)
        detections = self._nms_by_class(
            detections,
            iou_threshold=config.OWL_NMS_IOU
        )
        print(f"[OWL] После NMS: {len(detections)}")

        # Подсчёт
        counts = defaultdict(int)
        for (_, _, _, _, ru, _) in detections:
            counts[ru] += 1

        # Отрисовка
        annotated = frame.copy()
        for i, (x1, y1, x2, y2, ru, conf) in enumerate(detections, 1):
            color = _hash_color(ru)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{ru} {conf:.2f}"
            annotated = draw_text(annotated, label, (x1, max(0, y1 - 22)), 18,
                                  (255, 255, 255), color, fast=False)

        total = len(detections)
        title = f"OWLv2: {total} фруктов" if self.is_owlv2 else f"OWL-ViT: {total} фруктов"
        annotated = draw_text(annotated, title, (10, 10), 28,
                              (255, 255, 255), (150, 0, 100))

        print(f"[OWL] ✓ Итог: {total}  - {dict(counts)}")
        print(f"[OWL] ====== Конец ======\n")

        return annotated, {
            'total': total,
            'by_class': dict(counts),
            'frame_detections': total,
            'mode': 'detection',
            'engine': 'owlvit',
        }

# NMS по классам для OWL-ViT / OWLv2
    def _nms_by_class(self, detections, iou_threshold=0.3):
        """NMS по классам."""
        if not detections:
            return []
        # Группировка по классам
        by_class = defaultdict(list)
        for d in detections:
            by_class[d[4]].append(d)
        # Применяем NMS для каждого класса
        result = []
        for label, dets in by_class.items():
            if not dets:
                continue
            # Используем torchvision.ops.nms если доступно
            if torchvision_nms is not None and len(dets) > 1:
                try:
                    boxes_t = torch.tensor([d[:4] for d in dets], dtype=torch.float32)
                    scores_t = torch.tensor([d[5] for d in dets], dtype=torch.float32)
                    keep = torchvision_nms(boxes_t, scores_t, iou_threshold)
                    for idx in keep:
                        result.append(dets[idx])
                    continue
                except Exception:
                    pass
            # Fallback: простой NMS на Python
            dets = sorted(dets, key=lambda d: d[5], reverse=True)
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                dets = [d for d in dets
                        if self._iou(best[:4], d[:4]) < iou_threshold]
            result.extend(keep)
        return result

# Функция для вычисления IoU между двумя bounding box
    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0