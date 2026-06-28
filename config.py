# МОДЕЛИ
# YOLO модели (для камеры и быстрой обработки)
DETECTION_MODEL = 'yolov8m.pt'           # детекция (быстро, 3 фрукта COCO)
SEGMENTATION_MODEL = 'yolov8m-seg.pt'    # сегментация (маски)
CLASSIFICATION_MODEL = 'yolov8n-cls.pt'  # классификация (Top-5)

# OWL-ViT / OWLv2 модель (для точной детекции на фото)
# OWLv2 на +-30% точнее оригинального OWL-ViT
OWL_VIT_MODEL = 'google/owlv2-base-patch16-ensemble'
# Fallback (если OWLv2 не загрузится):
# OWL_VIT_MODEL = 'google/owlvit-base-patch32'


# ПАРАМЕТРЫ YOLO (для камеры  - быстро)
CONFIDENCE_THRESHOLD = 0.15   # минимальная уверенность (0.0 - 1.0)
IOU_THRESHOLD = 0.11          # порог NMS (ниже = меньше склеиваний)
IMAGE_SIZE = 1920             # разрешение (640, 1280, 1920)
MAX_DETECTIONS = 500          # максимум объектов на кадр


# ПАРАМЕТРЫ OWL-ViT (для фото  - точно, но медленно)
OWL_CONFIDENCE = 0.31          # минимальная уверенность (0.0 - 1.0)
OWL_MIN_AREA = 300             # минимум 300 px²
OWL_NMS_IOU = 0.33             # порог NMS (ниже = меньше склеиваний)
OWL_ASPECT_RATIO_MAX = 3.0     # максимальное соотношение сторон


# КЛАССЫ ФРУКТОВ
# YOLO COCO классы (только 3 фрукта)
FRUIT_CLASSES_COCO = {'apple', 'banana', 'orange'}

# Список фруктов для OWLv2 с PROMPT ENGINEERING
# Добавление "a photo of a X" даёт +-10% точности
FRUIT_LABELS_EN = [
    "a photo of an apple",
    "a photo of a banana",
    "a photo of an orange",
    "a photo of a pineapple",
    "a photo of a strawberry",
    "a photo of a lemon",
    "a photo of a lime",
    "a photo of a bunch of grapes",
    "a photo of a grapefruit",
    "a photo of a watermelon",
    "a photo of a melon",
    "a photo of a pear",
    "a photo of a peach",
    "a photo of an apricot",
    "a photo of a cherry",
    "a photo of a plum",
    "a photo of a pomegranate",
    "a photo of a fig",
    "a photo of a mango",
    "a photo of a papaya",
    "a photo of a guava",
    "a photo of a kiwi fruit",
    "a photo of an avocado",
    "a photo of a coconut",
    "a photo of a raspberry",
    "a photo of a blueberry",
    "a photo of a blackberry",
    "a photo of a cranberry",
    "a photo of a lychee",
    "a photo of a persimmon",
    "a photo of a quince",
    "a photo of a date fruit",
    "a photo of a kumquat",
    "a photo of a tangerine",
    "a photo of a mandarin orange",
    "a photo of a clementine",
    "a photo of a nectarine",
    "a photo of a dragon fruit",
    "a photo of a passion fruit",
    "a photo of a star fruit",
    "a photo of a durian",
    "a photo of a rambutan",
    "a photo of a longan",
    "a photo of a breadfruit",
    "a photo of a feijoa",
    "a photo of a tamarind",
    "a photo of a mulberry",
    "a photo of a gooseberry",
    "a photo of a currant",
    "a photo of an elderberry",
    "a photo of a boysenberry",
    "a photo of a cantaloupe",
    "a photo of a honeydew melon",
    "a photo of a plantain",
    "a photo of a pomelo",
    "a photo of a loquat",
    "a photo of a medlar",
    "a photo of a jujube",
    "a photo of a sapote",
    "a photo of a sapodilla",
    "a photo of a cherimoya",
    "a photo of a soursop",
    "a photo of a mangosteen",
    "a photo of a cacao pod",
    "a photo of a tomato",
    "a photo of a bell pepper",
    "a photo of a cucumber",
    "a photo of a zucchini",
    "a photo of an eggplant",
    "a photo of a head of cabbage",
    "a photo of a cauliflower",
    "a photo of a broccoli",
    "a photo of an ear of corn",
    "a photo of a mushroom",
]

# Словарь переводов (ключи без "a photo of "  - это чистые названия)
LABEL_TRANSLATION = {
    "apple": "Яблоко", "banana": "Банан", "orange": "Апельсин",
    "pineapple": "Ананас", "strawberry": "Клубника",
    "lemon": "Лимон", "lime": "Лайм",
    "grape": "Виноград", "bunch of grapes": "Виноград",
    "grapefruit": "Грейпфрут", "watermelon": "Арбуз",
    "melon": "Дыня", "pear": "Груша", "peach": "Персик",
    "apricot": "Абрикос", "cherry": "Вишня", "plum": "Слива",
    "pomegranate": "Гранат", "fig": "Инжир", "mango": "Манго",
    "papaya": "Папайя", "guava": "Гуава", "kiwi fruit": "Киви",
    "avocado": "Авокадо", "coconut": "Кокос", "raspberry": "Малина",
    "blueberry": "Черника", "blackberry": "Ежевика",
    "cranberry": "Клюква", "lychee": "Личи", "persimmon": "Хурма",
    "quince": "Айва", "date fruit": "Финик", "kumquat": "Кумкват",
    "tangerine": "Мандарин", "mandarin orange": "Мандарин",
    "clementine": "Клементин", "nectarine": "Нектарин",
    "dragon fruit": "Питайя", "passion fruit": "Маракуйя",
    "star fruit": "Карамбола", "durian": "Дуриан",
    "rambutan": "Рамбутан", "longan": "Лонган",
    "breadfruit": "Хлебное дерево", "feijoa": "Фейхоа",
    "tamarind": "Тамаринд", "mulberry": "Шелковица",
    "gooseberry": "Крыжовник", "currant": "Смородина",
    "elderberry": "Бузина", "boysenberry": "Бойзенова ягода",
    "cantaloupe": "Канталупа", "honeydew melon": "Медовая дыня",
    "plantain": "Плантан", "pomelo": "Помело", "loquat": "Мушмула",
    "medlar": "Мушмула", "jujube": "Зизифус", "sapote": "Сапота",
    "sapodilla": "Саподилла", "cherimoya": "Черимойя",
    "soursop": "Сметанное яблоко", "mangosteen": "Мангостан",
    "cacao pod": "Какао", "tomato": "Томат",
    "bell pepper": "Болгарский перец",
    "cucumber": "Огурец", "zucchini": "Кабачок", "eggplant": "Баклажан",
    "head of cabbage": "Капуста", "cauliflower": "Цветная капуста",
    "broccoli": "Брокколи", "ear of corn": "Кукуруза",
    "mushroom": "Гриб",
}

# TODO: добавить поддержку перевода с помощью Google Translate API (для неизвестных фруктов)
def translate_label(name):
    """
    Универсальный перевод с поддержкой "a photo of ..." и артиклей.
    
    Args:
        name: например "a photo of an apple" или "apple" или "an apple"
    
    Returns:
        Русское название, например "Яблоко"
    """
    if not name:
        return name
    
    lower = name.lower().strip()
    
    # Убираем "a photo of " из начала (prompt engineering префикс)
    if lower.startswith("a photo of "):
        lower = lower[len("a photo of "):]
    # Убираем артикли
    if lower.startswith("an "):
        lower = lower[3:]
    elif lower.startswith("a "):
        lower = lower[2:]
    
    lower = lower.strip()
    
    # Проверяем словарь
    if lower in LABEL_TRANSLATION:
        return LABEL_TRANSLATION[lower]
    
    # Fallback для COCO
    coco_mapping = {
        'apple': 'Яблоко',
        'banana': 'Банан',
        'orange': 'Апельсин',
    }
    if lower in coco_mapping:
        return coco_mapping[lower]
    
    return lower.capitalize()



# Flask настройки 
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB


# Пути к папкам и файлам
UPLOAD_DIR = 'static/uploads'
RESULT_DIR = 'static/results'
REPORTS_DIR = 'reports'
DB_PATH = 'history.db'


# Дополнительные настройки
CAMERA_WARMUP_FRAMES = 5    # количество кадров для ожидание камеры
MJPEG_QUALITY_RAW = 80      # качество MJPEG для сырых изображений
MJPEG_QUALITY_AI = 85       # качество MJPEG для изображений с AI-аннотациями
FRAME_TIMEOUT = 1.0         # таймаут между кадрами (секунды)