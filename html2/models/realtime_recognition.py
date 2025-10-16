# 即時辨識5 - 單幀版本，支援節點、中文翻譯與順暢關閉攝影機
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

# --------------------------
# 載入模型與標籤
# --------------------------
model = tf.keras.models.load_model("transformer_model.h5")
classes = np.load("label_classes.npy", allow_pickle=True)

# --------------------------
# Mediapipe 初始化
# --------------------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)
face_mesh = mp_face.FaceMesh(max_num_faces=1)

# --------------------------
# 開啟攝影機
# --------------------------
cap = cv2.VideoCapture(0)

# --------------------------
# 中文字型設定
# --------------------------
font_path = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體，確保存在
font = ImageFont.truetype(font_path, 32)

frame_count = 0

try:
    while True:
        ret, frame = c
        if not ret:
            break
        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測手部與臉部
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

        # 取得臉部特徵點
        face_x, face_y, face_z = [], [], []
        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                for i in range(68):
                    lm = face.landmark[i]
                    face_x.append(lm.x)
                    face_y.append(lm.y)
                    face_z.append(lm.z)
            cx, cy, cz = np.mean(face_x), np.mean(face_y), np.mean(face_z)
        else:
            cx, cy, cz = 0, 0, 0

        # 建立 keypoints 向量
        keypoints = []

        # 手部關鍵點
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                for lm in hand.landmark:
                    keypoints.extend([lm.x - cx, lm.y - cy, lm.z - cz])
        else:
            keypoints.extend([0] * 21 * 3 * 2)

        # 臉部關鍵點
        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face, mp_face.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
        for i in range(68):
            if i < len(face_x):
                keypoints.extend([face_x[i]-cx, face_y[i]-cy, face_z[i]-cz])
            else:
                keypoints.extend([0,0,0])

        # --------------------------
        # 模型預測（單幀）
        # --------------------------
        label = ""
        expected_length = 21*3*2 + 68*3
        if len(keypoints) == expected_length:
            x_input = np.array(keypoints).reshape(1, 1, -1)
            pred = model.predict(x_input, verbose=0)
            label = classes[np.argmax(pred)]

        # --------------------------
        # 用 PIL 畫中文與 frame 計數
        # --------------------------
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f"Frame: {frame_count}", font=font, fill=(255, 0, 0))
        draw.text((10, 70), f"翻譯結果: {label}", font=font, fill=(0, 255, 0))
        frame = np.array(img_pil)

        # 顯示影像
        cv2.imshow("SignLang AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --------------------------
    # 釋放資源
    # --------------------------
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()
    print("✅ 已順利關閉攝影機與視窗")
