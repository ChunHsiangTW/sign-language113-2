import os
import time
import csv
import cv2
import dlib
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# === 啟動檔案選擇視窗 ===
tk.Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="選擇影片檔案",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    print("未選擇影片")
    exit()

if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("❌ 找不到 shape_predictor_68_face_landmarks.dat")
    exit()


def get_rotation_metadata(video_path):
    try:
        import ffmpeg
        probe = ffmpeg.probe(video_path)
        rotate_tag = probe['streams'][0]['tags'].get('rotate', '0')
        return int(rotate_tag)
    except:
        return 0

rotation_code = get_rotation_metadata(video_path)

# === 初始化模型 ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === 影片初始化 ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 影片無法開啟")
    exit()

frame_idx = 0
coordinates_data = []
labels_data = []
start_frame = None

# === 開始處理每一幀 ===
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1000, 750))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === 臉部偵測（dlib） ===
        faces = detector(gray)
        face_coords = [(0.0, 0.0, 0.0)] * 68
        for face in faces:
            landmarks = predictor(gray, face)
            face_coords = [(landmarks.part(i).x, landmarks.part(i).y, 0.0) for i in range(68)]
            for i, (x, y, _) in enumerate(face_coords):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        # === 手部偵測（MediaPipe） ===
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        R_hand = [(0.0, 0.0, 0.0)] * 21
        L_hand = [(0.0, 0.0, 0.0)] * 21
        handedness = []

        if results.multi_handedness:
            for idx, hand_info in enumerate(results.multi_handedness):
                label = hand_info.classification[0].label
                handedness.append(label)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                hand_pts = [(int(x * frame.shape[1]), int(y * frame.shape[0]), z) for x, y, z in hand_pts]
                if handedness[idx] == 'Right':
                    R_hand = hand_pts
                else:
                    L_hand = hand_pts
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === 儲存每一幀的資料 ===
        flat_data = [frame_idx]
        for pt in R_hand + L_hand + face_coords:
            flat_data.extend(pt)
        coordinates_data.append(flat_data)

        # === 顯示畫面與控制 ===
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Landmark Annotator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            start_frame = frame_idx
            print(f"[起始] 記錄開始於 frame {start_frame}")
        elif key == ord('e') and start_frame is not None:
            end_frame = frame_idx
            label = input("輸入標註詞語：")
            labels_data.append([start_frame, end_frame, label])
            print(f"[完成] {label}：{start_frame} ~ {end_frame}")
            start_frame = None

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# === 儲存 CSV 檔案 ===
# 👉 儲存座標檔（coordinates.csv）
coord_header = ["frame"]
for i in range(21):
    coord_header += [f"R{i}_x", f"R{i}_y", f"R{i}_z"]
for i in range(21):
    coord_header += [f"L{i}_x", f"L{i}_y", f"L{i}_z"]
for i in range(68):
    coord_header += [f"F{i+1}_x", f"F{i+1}_y", f"F{i+1}_z"]

with open("coordinates.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(coord_header)
    writer.writerows(coordinates_data)

# 👉 儲存標註詞語檔（labels.csv）
with open("labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["start_frame", "end_frame", "label"])
    writer.writerows(labels_data)

print("✅ 已儲存 coordinates.csv 與 labels.csv")
