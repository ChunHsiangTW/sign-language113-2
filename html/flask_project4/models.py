from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)  # 帳號（姓名）
    password = db.Column(db.String(150), nullable=False)  # 密碼
    gender = db.Column(db.String(10), nullable=False)  # 性別
    age = db.Column(db.Integer, nullable=False)  # 年齡
    created_at = db.Column(db.TIMESTAMP, default=db.func.current_timestamp())  # 新增創建時間欄位
