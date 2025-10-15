import os  
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from PIL import ImageFont, ImageDraw, Image

# === 模型與資料路徑 ===
MODEL_PATH = "models/transformer_3d.keras"
LABEL_MAP_PATH = "data/label_map.npy"

MAX_FRAMES = 200
FEATURES = 21 * 3 * 2 + 68 * 3  # 雙手 + 臉部68點

# === 字型設定 (Windows 範例: 微軟正黑體) ===
font_path = "C:/Windows/Fonts/msjh.ttc"
font = ImageFont.truetype(font_path, 32)

# === 自訂層：Positional Encoding ===
class PositionalEncoding(layers.Layer):
    def __init__(self, length, dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        pos = np.arange(length)[:, None]
        i = np.arange(dim)[None, :]
        angle = pos / np.power(10000, (2 * (i // 2)) / dim)
        pe = np.zeros_like(angle)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

# === 載入模型與標籤 ===
with tf.keras.utils.custom_object_scope({'PositionalEncoding': PositionalEncoding}):
    model = load_model(MODEL_PATH)

label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
inv_map = {v: k for k, v in label_map.items()}

# === 初始化 Mediapipe ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2)
face_mesh = mp_face.FaceMesh(max_num_faces=1)

# === 開啟攝影機 ===
cap = cv2.VideoCapture(0)
buff = []
print("🎥 開始即時辨識（按 Q 離開）")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # --- 計算臉部中心 ---
    face_center = np.zeros(3)
    face_landmarks = None
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:68]])
        face_center = coords.mean(axis=0)

    # --- 取得雙手特徵 (固定長度 126) ---
    hand_points = np.zeros(21*3*2, dtype=np.float32)
    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            if idx >= 2:
                break
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            hand_points[idx*21*3:(idx+1)*21*3] = coords - np.tile(face_center, 21)

    # --- 取得臉部特徵 (固定長度 204) ---
    face_points = np.zeros(68*3, dtype=np.float32)
    if face_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark[:68]]).flatten()
        face_points = coords - np.tile(face_center, 68)

    # --- 合併特徵 + 標準化 ---
    frame_features = np.concatenate([hand_points, face_points])
    arr3 = frame_features.reshape(-1, 3)
    mean, std = arr3.mean(axis=0), arr3.std(axis=0)
    std[std == 0] = 1
    arr3 = (arr3 - mean) / std
    frame_features = arr3.flatten()

    buff.append(frame_features)
    if len(buff) > MAX_FRAMES:
        buff.pop(0)

    # --- 推論與顯示結果 ---
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    if len(buff) == MAX_FRAMES:
        X = np.expand_dims(np.array(buff, dtype=np.float32), 0)
        pred = model.predict(X, verbose=0)
        confidence = np.max(pred)
        if confidence > 0.5:
            label = inv_map[int(np.argmax(pred))]
        else:
            label = "沒結果"
        draw.text((30, 30), f"{label} ({confidence:.2f})", font=font, fill=(0, 255, 0))
    else:
        draw.text((30, 30), "沒結果", font=font, fill=(0, 255, 0))

    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- 畫出手部節點 (黑色) ---
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=2)
            )

    # --- 畫出臉部網格 (黑色) ---
    if face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
            mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1),
            mp.solutions.drawing_utils.DrawingSpec(color=(0,0,0), thickness=1)
        )

    # --- 顯示畫面 ---
    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
