import sqlite3
conn = sqlite3.connect("chatbot.db")
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(checkpoints)")
print(cursor.fetchall())