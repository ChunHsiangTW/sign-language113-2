from flask import Flask, render_template, request, redirect, url_for, flash, Response, Blueprint, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import mysql.connector
import re
import cv2
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
camera_bp = Blueprint('camera_bp', __name__)

# Flask-Login 初始化
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("已登出")
    return redirect(url_for('login'))

# 學習手語
@app.route("/learn")
@login_required
def learn():
    category = request.args.get("category")
    keyword = request.args.get("keyword", "").strip()

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    base_sql = """
    SELECT videos_new.*, images.filename AS image_filename
    FROM videos_new
    LEFT JOIN images ON videos_new.id = images.video_id
    """

    params = []
    conditions = []

    if category and category != "all":
        conditions.append("videos_new.category = %s")
        params.append(category)

    if keyword:
        conditions.append("(videos_new.title_cn LIKE %s OR videos_new.description LIKE %s OR videos_new.title_en LIKE %s)")
        kw_like = f"%{keyword}%"
        params.extend([kw_like, kw_like, kw_like])

    conditions.append("images.filename IS NOT NULL")

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

@app.route('/ranking') 
@login_required
def ranking(): return render_template('ranking.html')

@app.route('/daily-learning') 
@login_required
def daily_learning(): return render_template('daily_learning.html')

@app.route('/course_progress')
@login_required
def course_progress():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 取出所有影片基本資料 + 用戶學習進度（如果有）
    sql = """
    SELECT v.id, v.title_cn, v.title_en, v.category, v.description, v.filename,
           up.progress, up.completed
    FROM videos_new v
    LEFT JOIN user_progress up ON v.id = up.video_id AND up.user_id = %s
    ORDER BY v.id
    """
    cursor.execute(sql, (current_user.id,))
    courses = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('course_progress.html', courses=courses)


@app.route('/update_progress', methods=['POST'])
@login_required
def update_progress():
    data = request.json
    video_id = data.get('video_id')
    progress = data.get('progress')  # 0~100
    completed = data.get('completed', False)

    if not video_id or progress is None:
        return jsonify({'success': False, 'message': '缺少必要參數'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 檢查是否已有紀錄
    cursor.execute("SELECT id FROM user_progress WHERE user_id=%s AND video_id=%s", (current_user.id, video_id))
    record = cursor.fetchone()

    if record:
        # 更新
        cursor.execute("""
            UPDATE user_progress
            SET progress=%s, completed=%s, updated_at=NOW()
            WHERE user_id=%s AND video_id=%s
        """, (progress, completed, current_user.id, video_id))
    else:
        # 新增
        cursor.execute("""
            INSERT INTO user_progress (user_id, video_id, progress, completed)
            VALUES (%s, %s, %s, %s)
        """, (current_user.id, video_id, progress, completed))

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'success': True})


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
