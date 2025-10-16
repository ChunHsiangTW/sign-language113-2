import mysql.connector

def get_db_connection():
    conn = mysql.connector.connect(
        host="140.131.114.242",
        user="Handlang",           
        password="Hh114411@",
        database="114-Handlang",
        auth_plugin='mysql_native_password',
        charset='utf8mb4'
    )
    return conn