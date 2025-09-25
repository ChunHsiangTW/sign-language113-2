from flask import Flask, render_template, request, redirect, url_for, flash 
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from sqlalchemy.exc import IntegrityError

# 其他代碼


# 初始化 Flask 應用
app = Flask(__name__)

# 設定應用程式的配置
app.config['SECRET_KEY'] = 'your_secret_key'  # 用來加密會話的密鑰
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 禁止追蹤修改，減少警告

# 初始化資料庫和遷移工具
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 初始化 Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 定義 User 模型，包含帳號、密碼、性別、年齡
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    gender = db.Column(db.String(10), nullable=False)  # 新增性別欄位
    age = db.Column(db.Integer, nullable=False)  # 新增年齡欄位

# 用戶加載回調函數
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        gender = request.form['gender']
        age = request.form['age']

        # 檢查密碼是否符合要求
        if len(password) < 6 or not any(char.isdigit() for char in password) or not any(char.isalpha() for char in password):
            flash('密碼必須包含字母和數字，並且至少6個字符長！', 'danger')
            return redirect(url_for('register'))

        # 使用預設的密碼加密方法
        hashed_password = generate_password_hash(password)

        # 檢查是否已有此帳號、姓名和年齡相同的用戶
        existing_user = User.query.filter_by(username=username, gender=gender, age=age).first()

        if existing_user:
            # 如果帳號、姓名、年齡相同，則檢查是否密碼不同
            if check_password_hash(existing_user.password, password):
                flash('此帳號已註冊過，請選擇其他帳號名稱或更新密碼！', 'danger')
            else:
                # 如果密碼不同，則更新密碼
                existing_user.password = hashed_password
                db.session.commit()
                flash('密碼已更新！', 'success')
                return redirect(url_for('login'))
        else:
            # 如果沒有相同帳號、姓名、年齡的用戶，則可以註冊新用戶
            new_user = User(username=username, password=hashed_password, gender=gender, age=age)
            db.session.add(new_user)
            try:
                db.session.commit()
                flash('註冊成功，請登入！', 'success')
                return redirect(url_for('login'))
            except IntegrityError:
                # 這裡處理 unique constraint 錯誤
                db.session.rollback()
                flash('此帳號已註冊過，請選擇其他帳號名稱！', 'danger')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 查詢使用者是否存在
        user = User.query.filter_by(username=username).first()
        
        if user is None:
            # 使用者未註冊
            flash('尚未註冊，請先註冊！', 'danger')
        elif not check_password_hash(user.password, password):
            # 密碼錯誤
            flash('密碼錯誤，請重新輸入！', 'danger')
        else:
            # 登入成功
            login_user(user)
            flash('登入成功！', 'success')
            return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('已登出', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)


@app.route('/users')
@login_required
def users():
    all_users = User.query.all()  # 查詢所有使用者
    return render_template('users.html', users=all_users)

if __name__ == '__main__':
    app.run(debug=True)
