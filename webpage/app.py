from flask import Flask, render_template, request, redirect, url_for, flash, Response, Blueprint, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import mysql.connector
import re
import cv2
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_fallback_secret_key')
camera_bp = Blueprint('camera_bp', __name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '11146038@gmail.com'   # ⚠️ 改成你自己的 Gmail
app.config['MAIL_PASSWORD'] = 'sofe qvyu zpat fgws'           # ⚠️ 要用「應用程式密碼」不是登入密碼
app.config['MAIL_DEFAULT_SENDER'] = ('HandLang 手語通', app.config['MAIL_USERNAME'])
mail = Mail(app)
s = URLSafeTimedSerializer(app.secret_key)

# Flask-Login 初始化
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "💡 請先登入以繼續使用此頁面 "

# 資料庫連線函式
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='123imd',
        database='hand_project',
        charset='utf8mb4'
    )


def course_progress():
    cnx = get_db_connection()   # 取得連線
    cursor = cnx.cursor()       # 從連線建立游標
    
    sql = """
    SELECT v.title_cn, up.progress, up.completed
    FROM user_progress up
    JOIN videos_new v ON up.video_id = v.id
    WHERE up.user_id = %s
    """
    
    cursor.execute(sql, (current_user.id,))
    results = cursor.fetchall()
    
    cursor.close()
    cnx.close()
    
    # 你可以接著把 results 傳給模板或處理
    return results


# 使用者模型
class User(UserMixin):
    def __init__(self, id_, email, name, password_hash):
        self.id = id_
        self.email = email
        self.name = name
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            return User(user['id'], user['email'], user['name'], user['password'])
        return None

    @staticmethod
    def get_by_email(email):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            return User(user['id'], user['email'], user['name'], user['password'])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# 相機串流
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@camera_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.register_blueprint(camera_bp)

# 影片上傳設定
UPLOAD_FOLDER = 'static/videos_new'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 首頁
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/base')
def base():
    return render_template('base.html')

# 註冊
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        password = request.form['password']
        confirm = request.form['confirm']

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("請輸入有效的 Email")
            return redirect(url_for('register'))

        if len(password) < 6 or not re.search(r'[A-Za-z]', password) or not re.search(r'[0-9]', password):
            flash("密碼至少 6 碼，並含有英文與數字")
            return redirect(url_for('register'))

        if password != confirm:
            flash("兩次輸入的密碼不一致")
            return redirect(url_for('register'))

        try:
            hashed_password = generate_password_hash(password)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (email, name, gender, age, password) VALUES (%s, %s, %s, %s, %s)",
                           (email, name, gender, age, hashed_password))
            conn.commit()
            cursor.close()
            conn.close()
            flash("註冊成功，請登入！")
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash("這個 Email 已被註冊")
            return redirect(url_for('register'))

    return render_template('register.html')

# 登入
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.get_by_email(email)

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash("登入成功！")
            return redirect(url_for('learn'))
        else:
            flash("登入失敗，請檢查帳號密碼")
            return redirect(url_for('login'))

    return render_template('login.html')

# 登出
@app.route("/logout")
@login_required
def logout():
    logout_user()  # 清除登入狀態
    return render_template("logout.html")

# 修改個人資料
@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        # 取得表單資料
        name = request.form.get('name')
        gender = request.form.get('gender')
        age = request.form.get('age')

        # 更新資料庫
        try:
            cursor.execute(
                "UPDATE users SET name = %s, gender = %s, age = %s WHERE id = %s",
                (name, gender, age, current_user.id)
            )
            conn.commit()
            flash("個人資料已更新成功！")
            return redirect(url_for('profile'))
        except Exception as e:
            print("更新錯誤:", e)
            flash("更新失敗，請稍後再試")
            return redirect(url_for('edit_profile'))
        finally:
            cursor.close()
            conn.close()

    # GET 請求顯示目前資料
    cursor.execute("SELECT * FROM users WHERE id = %s", (current_user.id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return render_template('edit_profile.html', user=user)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        # 嘗試解 token（有效期 1800 秒 = 30 分鐘）
        email = s.loads(token, salt='password-reset-salt', max_age=1800)
    except SignatureExpired:
        return render_template('reset_password.html', message="重設連結已過期，請重新申請。")
    except BadSignature:
        return render_template('reset_password.html', message="重設連結無效，請確認連結是否正確。")

    message = None
    if request.method == 'POST':
        new_password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm', '').strip()

        # 基本驗證（可擴充）
        if not new_password or len(new_password) < 6:
            message = "密碼需至少 6 碼。"
            return render_template('reset_password.html', message=message)
        if new_password != confirm:
            message = "兩次密碼輸入不一致。"
            return render_template('reset_password.html', message=message)

        # hash 密碼後寫入 MySQL
        hashed = generate_password_hash(new_password)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed, email))
            conn.commit()
            cursor.close()
            conn.close()
            message = "密碼已更新成功！請使用新密碼登入。"
            return render_template('reset_password.html', message=message)
        except Exception as e:
            print("更新密碼錯誤：", e)
            message = "更新密碼失敗，請稍後再試。"

    return render_template('reset_password.html', message=message)


# 學習手語
@app.route("/learn")
@login_required
def learn():
    category = request.args.get("category")
    keyword = request.args.get("keyword", "").strip()

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 直接從 videos_new 撈資料
    base_sql = "SELECT * FROM videos_new"
    params = []
    conditions = []

    # 分類條件
    if category and category != "all":
        conditions.append("category = %s")
        params.append(category)

    # 搜尋條件
    if keyword:
        conditions.append("(title_cn LIKE %s OR description LIKE %s OR title_en LIKE %s OR keywords LIKE %s)")
        kw_like = f"%{keyword}%"
        params.extend([kw_like, kw_like, kw_like, kw_like])

    # 如果有條件，加入 WHERE
    if conditions:
        base_sql += " WHERE " + " AND ".join(conditions)

    cursor.execute(base_sql, params)
    lessons = cursor.fetchall()
    conn.close()

    return render_template("learn.html", lessons=lessons, category=category, keyword=keyword)

# 單一影片播放
@app.route("/video/<title_en>")
@login_required
def video(title_en):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT v.*, i.filename AS image_filename
            FROM videos_new v LEFT JOIN images i ON v.id = i.video_id
            WHERE v.title_en = %s
        """, (title_en,))
        lesson = cursor.fetchone()

        if not lesson:
            flash("找不到該影片資料")
            return redirect(url_for("learn"))

        current_id = lesson['id']

        cursor.execute("SELECT title_en FROM videos_new WHERE id < %s ORDER BY id DESC LIMIT 1", (current_id,))
        prev_video = cursor.fetchone()

        cursor.execute("SELECT title_en FROM videos_new WHERE id > %s ORDER BY id ASC LIMIT 1", (current_id,))
        next_video = cursor.fetchone()

        if not prev_video:
            cursor.execute("SELECT title_en FROM videos_new ORDER BY id DESC LIMIT 1")
            prev_video = cursor.fetchone()

        if not next_video:
            cursor.execute("SELECT title_en FROM videos_new ORDER BY id ASC LIMIT 1")
            next_video = cursor.fetchone()

    except Exception as e:
        print(f"查詢錯誤：{e}")
        flash("系統錯誤，無法載入資料")
        return redirect(url_for("learn"))
    finally:
        conn.close()

    return render_template("video.html", lesson=lesson,
                           prev_title_en=prev_video['title_en'],
                           next_title_en=next_video['title_en'])

# 個人資料
@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (current_user.id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template('profile.html', user=user)

# 影片上傳
@app.route('/upload_video', methods=['GET', 'POST'])
@login_required
def upload_video():
    if request.method == 'POST':
        title = request.form.get('title')
        file = request.files.get('video')

        # 檢查標題與檔案是否有填寫/上傳
        if not title or not file or file.filename == '':
            flash('請輸入標題並選擇影片檔案')
            return redirect(request.url)

        # 檢查檔案格式是否允許
        if not allowed_file(file.filename):
            flash('不支援的檔案格式，請上傳影片檔案(mp4, avi, mov, mkv)')
            return redirect(request.url)

        # 取得安全檔名並儲存檔案
        filename = secure_filename(file.filename)
        save_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, filename)
        file.save(save_path)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 寫入資料庫
            cursor.execute(
                "INSERT INTO videos_new (title_en, filename) VALUES (%s, %s)",
                (title, filename)
            )
            conn.commit()
            cursor.close()
            conn.close()

            flash('影片上傳成功！')
            return redirect(url_for('upload_video'))

        except Exception as e:
            print('資料庫錯誤:', e)
            flash('資料庫錯誤，請稍後再試')
            return redirect(request.url)

    # GET 請求回傳上傳頁面
    return render_template('upload_video.html')

# 其他頁面
@app.route('/about')
def about(): return render_template('about.html')

@app.route('/practice') 
@login_required
def practice(): return render_template('practice.html')
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    message = None
    if request.method == "POST":
        email = request.form.get("email")
        conn = sqlite3.connect("hand_project.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user:
            # 寄出重設密碼郵件
            try:
                msg = Message(
                    subject="🔑 手語通 - 密碼重設通知",
                    recipients=[email],
                    body=f"您好，這是您的密碼重設連結：\n\nhttp://127.0.0.1:5000/reset_password?email={email}\n\n若非您本人操作，請忽略此信件。"
                )
                mail.send(msg)
                message = f"已寄出密碼重設信至 {email}，請至信箱確認。"
            except Exception as e:
                message = f"寄信失敗：{str(e)}"
        else:
            message = "查無此 Email，請確認輸入是否正確。"

    return render_template("forgot_password.html", message=message)

@app.route('/daily-learning') 
@login_required
def daily_learning(): return render_template('daily_learning.html')


@app.route('/game_mode') 
@login_required
def game_mode(): return render_template('game_mode.html')

@app.route('/achievement') 
@login_required
def achievement(): return render_template('achievement.html')

@app.route('/realtime_sign') 
@login_required
def realtime_sign(): return render_template('realtime_sign.html')

@app.route('/history') 
@login_required
def history(): return render_template('history.html')

@app.route('/community') 
@login_required
def community(): return render_template('community.html')

@app.route('/group_tasks')
@login_required
def group_tasks(): return render_template('group_tasks.html')

@app.route('/settings') 
@login_required
def settings(): return render_template('settings.html')

@app.route('/notifications') 
@login_required
def notifications(): return render_template('notifications.html')

@app.route('/admin_dashboard') 
@login_required
def admin_dashboard(): return render_template('admin_dashboard.html')

@app.route('/statistics') 
@login_required
def statistics():
    return render_template('statistics.html')





if __name__ == '__main__':
    app.run(debug=True)
