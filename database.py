import sqlite3, json
from datetime import datetime
import config

# Инициализация базы данных (создание таблицы, если её нет)
def init_db():
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, filename TEXT,
        total_fruits INTEGER, by_class TEXT,
        result_image TEXT, mode TEXT DEFAULT 'detection',
        engine TEXT DEFAULT 'yolo'
    )''')
    cur.execute("PRAGMA table_info(requests)")
    cols = [r[1] for r in cur.fetchall()]
    
    # Миграции для старых БД (todo можно сделать более аккуратно remove this comment)
    # if 'mode' not in cols:
    #     cur.execute("ALTER TABLE requests ADD COLUMN mode TEXT DEFAULT 'detection'")
    # if 'engine' not in cols:
    #     cur.execute("ALTER TABLE requests ADD COLUMN engine TEXT DEFAULT 'yolo'")
    conn.commit()
    conn.close()

# save_request сохраняет информацию о запросе в базу данных
def save_request(filename, stats, result_image):
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute('''INSERT INTO requests
        (timestamp, filename, total_fruits, by_class, result_image, mode, engine)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (datetime.now().isoformat(), filename, stats.get('total', 0),
         json.dumps(stats.get('by_class', {}), ensure_ascii=False),
         result_image, stats.get('mode', 'detection'),
         stats.get('engine', 'yolo')))  # НОВОЕ: сохраняем движок
    conn.commit()
    conn.close()

# get_all_requests возвращает все запросы из базы данных в виде списка словарей
def get_all_requests():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM requests ORDER BY id DESC').fetchall()
    conn.close()
    result = []
    for r in rows:
        row = dict(r)
        try: row['by_class'] = json.loads(row['by_class'])
        except: pass
        result.append(row)
    return result

# clear_history удаляет все записи из таблицы requests
def clear_history():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute('DELETE FROM requests')
    conn.commit()
    conn.close()