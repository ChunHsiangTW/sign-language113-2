# 檔名: app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import re  # ✅ 修正：要用正則驗證 email

# ---------- 資料庫設定 ----------
DB_CONFIG = {
    'host': "140.131.114.242",
    'user': "Handlang",
    'password': "Hh114411@",
    'database': "114-Handlang",
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def get_db():
    return pymysql.connect(**DB_CONFIG)

# ---------- Flask App ----------
app = Flask(__name__)
app.secret_key = '114411'  # ✅ 一定要有這行，session 才能正常工作

# ---------- 管理員權限 ----------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('user_id') != 1:
            return "<h1>權限不足</h1><p>只有管理員才能進入此頁面。</p><a href='/'>回首頁</a>", 403
        return f(*args, **kwargs)
    return decorated_function

# ---------- 首頁 ----------
@app.route('/')
def index():
    return render_template('index.html', user_name=session.get('user_name'))

@app.route('/translate')
def translate():
    return render_template('translate.html')

@app.route('/flashcards')
def flashcards():
    return render_template('flashcards.html', user_name=session.get('user_name'))

# ---------- 管理頁面 ----------
@app.route('/admin')
@admin_required
def admin():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, email, name FROM users ORDER BY id")
    users = cur.fetchall()
    cur.execute("SELECT * FROM flashcards ORDER BY id")
    cards = cur.fetchall()
    conn.close()
    return render_template('admin.html', users=users, cards=cards)

@app.route('/add_card', methods=['POST'])
@admin_required
def add_card():
    title = request.form.get('title')
    thumb = request.form.get('thumb')
    video = request.form.get('video')
    desc = request.form.get('desc')

    if not all([title, thumb, video]):
        flash("請完整填寫所有欄位", "error")
        return redirect(url_for('admin'))

    thumb_path = f"thumbs/{thumb}"
    video_path = f"videos/{video}"

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO flashcards (title, thumb, video, `desc`) VALUES (%s, %s, %s, %s)",
            (title, thumb_path, video_path, desc)
        )
        conn.commit()
        flash("新增成功", "success")
    except Exception as e:
        flash(f"新增失敗: {str(e)}", "error")
    finally:
        conn.close()

    return redirect(url_for('admin'))

# ---------- 登入註冊頁 ----------
@app.route('/auth')
def auth():
    return render_template('auth.html')

# ---------- 註冊 ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        name = request.form.get('name', '').strip()
        gender = request.form.get('gender', '')
        age = request.form.get('age', '')
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        # 檢查欄位是否完整
        if not all([email, name, gender, age, password, confirm]):
            flash("請完整填寫所有欄位", "error")
            return redirect(url_for('register'))

        # 驗證 Email 格式
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Email 格式不正確", "error")
            return redirect(url_for('register'))

        # 密碼確認
        if password != confirm:
            flash("兩次密碼不一致", "error")
            return redirect(url_for('register'))

        # 加密密碼
        hashed = generate_password_hash(password)

        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (email, name, gender, age, password) VALUES (%s,%s,%s,%s,%s)",
                (email, name, gender, age, hashed)
            )
            conn.commit()
            flash("註冊成功！請登入", "success")
            return redirect(url_for('auth'))
        except pymysql.err.IntegrityError:
            flash("Email 已被註冊", "error")
            return redirect(url_for('register'))
        finally:
            conn.close()

    # GET → 顯示註冊頁面
    return render_template('register.html')

# ---------- 登入 ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash("登入成功", "success")
            return redirect(url_for('index'))
        else:
            flash("帳號或密碼錯誤", "error")
            return redirect(url_for('auth'))

    # 如果只是 GET 進入 /login，就導回登入頁
    return redirect(url_for('auth'))

# ---------- 登出 ----------
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    flash("已登出。", "success")
    return redirect(url_for('auth'))

# ---------- API ----------
@app.route('/api/get_cards')
def get_cards():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM flashcards")
    cards = cur.fetchall()
    conn.close()

    return jsonify(cards)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
