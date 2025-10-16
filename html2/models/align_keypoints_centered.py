# 標註節點位置 - 安全版 align_keypoints_centered.py
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from tqdm import tqdm  # ✅ 顯示進度條（需先 pip install tqdm）

# === Mediapipe 初始化 ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

# === 路徑設定 ===
VIDEO_DIR = "static/datavideos"   # 🎥 影片資料夾
OUTPUT_DIR = "data"               # 📄 輸出CSV資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 建立 CSV 標題列 ===
header = ["filename", "frame", "label"]
for i in range(21 * 2 * 3):
    header.append(f"hand_{i}")
for i in range(68 * 3):
    header.append(f"face_{i}")

# === 處理所有影片 ===
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".MOV", ".mov"))]

print(f"🎬 共偵測到 {len(video_files)} 支影片，開始擷取關鍵點...\n")

for filename in video_files:
    label = os.path.splitext(filename)[0]  # 例如「不見1」
    output_file = os.path.join(OUTPUT_DIR, f"{label}_keypoints.csv")

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, filename))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 影片總幀數

    # tqdm 進度條
    for frame_idx in tqdm(range(total_frames), desc=f"處理 {filename}", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        # === 臉部中心 ===
        face_center = np.zeros(3)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            xs = [lm.x for lm in face_landmarks.landmark[:68]]
            ys = [lm.y for lm in face_landmarks.landmark[:68]]
            zs = [lm.z for lm in face_landmarks.landmark[:68]]
            face_center = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])

        # === 手部節點（以臉為中心對齊） ===
        hand_points = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    coord = np.array([lm.x, lm.y, lm.z]) - face_center
                    hand_points.extend(coord.tolist())
        else:
            hand_points = [0] * (21 * 3 * 2)
        if len(hand_points) < 21 * 3 * 2:
            hand_points.extend([0] * (21 * 3 * 2 - len(hand_points)))

        # === 臉部節點（以臉為中心對齊） ===
        face_points = []
        if face_results.multi_face_landmarks:
            for lm in face_landmarks.landmark[:68]:
                coord = np.array([lm.x, lm.y, lm.z]) - face_center
                face_points.extend(coord.tolist())
        else:
            face_points = [0] * (68 * 3)

        # === 寫入CSV（防止錯誤行）===
        row = [filename, frame_idx, label] + hand_points + face_points

        if len(row) == len(header):
            with open(output_file, mode="a", newline="") as f:
                csv.writer(f).writerow(row)
        else:
            print(f"⚠️ 跳過 {filename} 第 {frame_idx} 幀，欄位數異常（{len(row)} vs {len(header)}）")

    cap.release()
    print(f"✅ 完成影片 {filename} 的關鍵點擷取，儲存於 {output_file}\n")

print("🎉 全部影片的關鍵點擷取已完成！CSV 檔案都在 /data 資料夾中。")
