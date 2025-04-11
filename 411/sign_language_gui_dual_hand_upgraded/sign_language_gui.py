
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
    hand_gestures = json.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

sentence = []
last_time_seen = time.time()
instruction_text = "🫱🫲 雙手辨識已啟用"

def draw_text(frame, text, pos=(30, 30), font_size=28, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("msjh.ttc", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def update_frame():
    global sentence, last_time_seen
    ret, frame = cap.read()
    if not ret:
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    current_time = time.time()
    result_texts = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [lm.y for lm in hand_landmarks.landmark[:6]]
            center = np.mean(landmarks[:5])
            code = "".join(['1' if y < center else '0' for y in landmarks])
            result = hand_gestures.get(code, "未知手勢")
            if results.multi_handedness:
                hand_label = results.multi_handedness[i].classification[0].label
                side = "左手" if hand_label == "Left" else "右手"
                result_texts.append(f"{side}: {result}")
            if result != "未知手勢" and (len(sentence) == 0 or sentence[-1] != result):
                sentence.append(result)
                last_time_seen = current_time
    else:
        if current_time - last_time_seen > 1.5 and sentence:
            print("📝 組成句子：", " ".join(sentence))
            sentence.clear()

    frame = draw_text(frame, instruction_text, (30, 20))
    frame = draw_text(frame, " ".join(result_texts), (30, 60))
    frame = draw_text(frame, f"組句中：{' '.join(sentence)}", (30, 100))

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

def capture_hand_gesture():
    captured = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            if frame_count < 5:
                landmarks = [lm.y for lm in hand.landmark[:6]]
                captured.append(landmarks)
                frame_count += 1
                cv2.waitKey(1000)
            elif frame_count >= 5:
                break
        cv2.waitKey(100)
    if len(captured) == 5:
        avg = np.mean(captured, axis=0)
        center = np.mean(avg[:5])
        code = "".join(['1' if y < center else '0' for y in avg])
        return code
    return None

def add_gesture():
    label = simpledialog.askstring("新增手勢", "請輸入手勢對應的詞語：")
    if label:
        code = capture_hand_gesture()
        if code:
            hand_gestures[code] = label
            with open(GESTURES_FILE, "w", encoding="utf-8") as f:
                json.dump(hand_gestures, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("成功", f"✅ 手勢【{label}】已儲存！")
        else:
            messagebox.showerror("錯誤", "❌ 擷取失敗")

def delete_gesture():
    label = simpledialog.askstring("刪除手勢", "請輸入要刪除的詞語：")
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
        messagebox.showerror("未找到", "找不到該手勢")

# GUI 建立
root = tk.Tk()
root.title("雙手辨識組句 GUI")
root.geometry("740x560")

video_label = tk.Label(root)
video_label.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="➕ 新增手勢", command=add_gesture, bg="lightblue", font=("Arial", 12)).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="🗑️ 刪除手勢", command=delete_gesture, bg="tomato", font=("Arial", 12)).grid(row=0, column=1, padx=10)

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
