#1影片轉JSON
import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# === 基本設定 ===
VIDEO_DIR = "static/datavideos"   # 影片來源資料夾
OUTPUT_DIR = "data"               # 輸出JSON資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FRAMES = 200
FEATURES = 21 * 3 * 2 + 68 * 3  # 330維 (雙手+臉)

# === 初始化Mediapipe ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

# === 處理所有影片 ===
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))]
print(f"🎬 共發現 {len(video_files)} 支影片，開始處理...")

for video_name in video_files:
    label = os.path.splitext(video_name)[0]
    video_path = os.path.join(VIDEO_DIR, video_name)
    output_json = os.path.join(OUTPUT_DIR, f"{label}.json")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_data = []

    for fid in tqdm(range(total_frames), desc=f"處理 {video_name}", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        # === 臉中心 ===
        face_center = np.zeros(3)
        face_landmarks = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:68]])
            face_center = coords.mean(axis=0)

        # === 手部節點 ===
        hand_points = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                coords = coords - face_center
                hand_points.extend(coords.flatten().tolist())
        while len(hand_points) < 21 * 3 * 2:
            hand_points.extend([0.0] * (21 * 3))

        # === 臉部節點 ===
        face_points = []
        if face_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:68]])
            coords = coords - face_center
            face_points = coords.flatten().tolist()
        else:
            face_points = [0.0] * (68 * 3)

        # === 合併 ===
        frame_features = np.array(hand_points + face_points, dtype=np.float32)
        arr3 = frame_features.reshape(-1, 3)
        mean, std = arr3.mean(axis=0), arr3.std(axis=0)
        std[std == 0] = 1
        arr3 = (arr3 - mean) / std
        frames_data.append(arr3.flatten().tolist())

    cap.release()

    # === 補幀或裁切 ===
    if len(frames_data) < MAX_FRAMES:
        pad = MAX_FRAMES - len(frames_data)
        frames_data.extend([[0.0] * FEATURES] * pad)
    else:
        frames_data = frames_data[:MAX_FRAMES]

    # === 儲存 JSON ===
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"label": label, "frames": frames_data}, f, ensure_ascii=False, indent=2)
    print(f"✅ 已儲存：{output_json}")

print("🎉 所有影片已完成 3D 關鍵點擷取並以臉為中心平移！")
