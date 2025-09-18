import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont

GESTURES_FILE = "gestures.json"

if not os.path.exists(GESTURES_FILE):
    with open(GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=4, ensure_ascii=False)

with open(GESTURES_FILE, "r", encoding="utf-8") as f:
    try:
        hand_gestures = json.load(f)
    except json.JSONDecodeError:
        hand_gestures = {}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
sentence = []
last_time_seen = time.time()
instruction_text = "請點選按鈕新增、辨識或刪除手勢"

def draw_text(frame, text, pos=(30, 30), font_size=28, color=(0, 255, 0)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("msjh.ttc", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def update_frame():
    global instruction_text, sentence, last_time_seen
    ret, frame = cap.read()
    if not ret:
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    result_text = "未偵測"
    current_time = time.time()

    if recognizing and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [lm.y for lm in hand_landmarks.landmark[:6]]
            center = np.mean(landmarks[:5])
            code = "".join(['1' if y < center else '0' for y in landmarks])
            result_text = hand_gestures.get(code, "未知手勢")
            if result_text != "未知手勢" and (len(sentence) == 0 or sentence[-1] != result_text):
                sentence.append(result_text)
                last_time_seen = current_time
    elif recognizing:
        if current_time - last_time_seen > 1.5 and sentence:
            print("📝 句子辨識結果：", " ".join(sentence))
            sentence = []

    frame = draw_text(frame, f"{instruction_text}", (30, 20))
    frame = draw_text(frame, f"當前辨識：{result_text}", (30, 60))
    frame = draw_text(frame, f"組句中：{' '.join(sentence)}", (30, 100))

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

def capture_hand_gesture():
    captured = []
    started = False
    frame_count = 0
    global instruction_text
    instruction_text = "📸 偵測中，擺好姿勢..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            if not started:
                started = True
                frame_count = 0
                instruction_text = "✅ 偵測到手勢，開始拍攝..."

            if started and frame_count < 5:
                landmarks = [lm.y for lm in hand.landmark[:6]]
                captured.append(landmarks)
                frame_count += 1
                instruction_text = f"📷 拍攝中：{frame_count}/5"
                cv2.waitKey(1000)
            elif frame_count >= 5:
                break
        else:
            instruction_text = "請擺出你的手勢..."

        cv2.waitKey(100)

    instruction_text = "完成！可繼續新增、辨識或刪除"

    if len(captured) == 5:
        avg = np.mean(captured, axis=0)
        center = np.mean(avg[:5])
        code = "".join(['1' if y < center else '0' for y in avg])
        return code
    return None

def add_gesture():
    label = simpledialog.askstring("輸入手勢名稱", "這個手勢代表的詞語（例如：你好）：")
    if not label:
        return
    code = capture_hand_gesture()
    if code:
        hand_gestures[code] = label
        with open(GESTURES_FILE, "w", encoding="utf-8") as f:
            json.dump(hand_gestures, f, indent=4, ensure_ascii=False)
        messagebox.showinfo("成功", f"✅ 已新增手勢：{label}")
    else:
        messagebox.showerror("錯誤", "❌ 擷取失敗，請再試一次。")

def delete_gesture():
    label = simpledialog.askstring("刪除手勢", "請輸入要刪除的詞語（中文）：")
    if not label:
        return
    found = False
    for code, word in list(hand_gestures.items()):
        if word == label:
            del hand_gestures[code]
            found = True
            break
    if found:
        with open(GESTURES_FILE, "w", encoding="utf-8") as f:
            json.dump(hand_gestures, f, indent=4, ensure_ascii=False)
        messagebox.showinfo("刪除成功", f"✅ 已刪除手勢：{label}")
    else:
        messagebox.showerror("未找到", f"❌ 找不到手勢：{label}")

def toggle_recognition():
    global recognizing, instruction_text
    recognizing = not recognizing
    if recognizing:
        instruction_text = "🧠 手語辨識已啟動..."
        recog_button.config(text="🛑 停止辨識")
    else:
        instruction_text = "🔴 已停止辨識"
        recog_button.config(text="▶️ 開始辨識")

root = tk.Tk()
root.title("一體化手語系統 GUI（含刪除功能）")
root.geometry("700x580")

video_label = tk.Label(root)
video_label.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

add_button = tk.Button(button_frame, text="➕ 新增手勢", command=add_gesture, bg="lightblue", font=("Arial", 12))
add_button.grid(row=0, column=0, padx=10)

recog_button = tk.Button(button_frame, text="▶️ 開始辨識", command=toggle_recognition, bg="lightgreen", font=("Arial", 12))
recog_button.grid(row=0, column=1, padx=10)

del_button = tk.Button(button_frame, text="🗑️ 刪除手勢", command=delete_gesture, bg="tomato", font=("Arial", 12))
del_button.grid(row=0, column=2, padx=10)

recognizing = False
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
