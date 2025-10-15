import os
import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "140.131.114.242"),
        user=os.getenv("DB_USER", "Handlang"),
        password=os.getenv("DB_PASS", "Hh114411@"),
        database=os.getenv("DB_NAME", "114-Handlang"),
        auth_plugin='mysql_native_password',
        charset='utf8mb4'
    )
