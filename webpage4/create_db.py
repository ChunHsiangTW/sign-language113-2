import sqlite3
import os

# 建立資料庫連線
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# 建立 users 資料表（有 id 主鍵與 progress 欄位）
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    gender TEXT,
    age INTEGER,
    password TEXT NOT NULL,
    progress INTEGER DEFAULT 0
)
''')

conn.commit()
conn.close()

print("✅ users.db 資料庫建立完成！路徑：", os.path.abspath("users.db"))
