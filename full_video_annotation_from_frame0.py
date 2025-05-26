
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import os

# 初始化
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 選擇影片
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="選擇影片", filetypes=[("MP4 files", "*.mp4")])
if not video_path:
    print("未選擇影片，程式結束")
    exit()

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ✅ 強制從第0幀開始
fps = cap.get(cv2.CAP_PROP_FPS)
output_data = []
current_frame = 0
paused = False
target_fps = 10

print("🎬 播放從 frame 0 開始，直到影片結束")

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (700, 1000))
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        display_frame = frame.copy()

        info_panel = np.ones((h, 500, 3), dtype=np.uint8) * 255
        x_col_L, x_col_R, y_offset, spacing = 10, 260, 25, 18

        # 手部偵測與繪製
        hand_results = hands.process(rgb)
        frame_coords = np.zeros((42, 3))
        hand_labels = {}
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                label = hand_results.multi_handedness[idx].classification[0].label
                hand_labels[label] = hand_landmarks
                for i, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if label == "Right":
                        frame_coords[i] = [lm.x, lm.y, lm.z]
                        cv2.circle(display_frame, (cx, cy), 3, (0, 255, 0), -1)
                        cv2.putText(display_frame, f"R{i+1}", (cx+4, cy), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                    else:
                        frame_coords[i + 21] = [lm.x, lm.y, lm.z]
                        cv2.circle(display_frame, (cx, cy), 3, (255, 0, 0), -1)
                        cv2.putText(display_frame, f"L{i+1}", (cx+4, cy), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)

        # info panel 手部座標
        cv2.putText(info_panel, "Right Hand", (x_col_L, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(info_panel, "Left Hand", (x_col_R, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        y_offset += spacing
        for i in range(21):
            if "Right" in hand_labels:
                lm = hand_labels["Right"].landmark[i]
                txt = f"R{i+1}: {lm.x:.2f} {lm.y:.2f} {lm.z:.2f}"
                cv2.putText(info_panel, txt, (x_col_L, y_offset), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1)
            if "Left" in hand_labels:
                lm = hand_labels["Left"].landmark[i]
                txt = f"L{i+1}: {lm.x:.2f} {lm.y:.2f} {lm.z:.2f}"
                cv2.putText(info_panel, txt, (x_col_R, y_offset), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1)
            y_offset += spacing

        # 臉部偵測
        face_coords = np.full((68, 3), -1.0)
        face_results = face_mesh.process(rgb)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            y_offset += spacing
            cv2.putText(info_panel, "Face", (x_col_L, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            y_offset += spacing
            y_offset_L, y_offset_R = y_offset, y_offset
            for i, lm in enumerate(face_landmarks.landmark[:68]):
                face_coords[i] = [lm.x, lm.y, lm.z]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), -1)
                cv2.putText(display_frame, f"F{i+1}", (cx+3, cy), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
                txt = f"F{i+1}: {lm.x:.2f} {lm.y:.2f} {lm.z:.2f}"
                if i < 34:
                    cv2.putText(info_panel, txt, (x_col_L, y_offset_L), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1)
                    y_offset_L += spacing
                else:
                    cv2.putText(info_panel, txt, (x_col_R, y_offset_R), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1)
                    y_offset_R += spacing

        # 顯示畫面
        cv2.putText(display_frame, f"Frame: {current_frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        combined = np.hstack((display_frame, info_panel))
        cv2.imshow("手部＋臉部座標標註", combined)

        # 儲存
        row = [current_frame] + frame_coords.flatten().tolist() + face_coords.flatten().tolist()
        output_data.append(row)
        current_frame += 1

    # 控制鍵區（空白鍵、q、e）
    key = cv2.waitKey(int(1000 / target_fps) if not paused else 0) & 0xFF
    if key == ord(' '): paused = not paused
    elif key == ord('q') or key == 27: break
    elif key == ord('e'):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        current_frame = total_frames - 1
    if cv2.getWindowProperty("手部＋臉部座標標註", cv2.WND_PROP_VISIBLE) < 1: break

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()

# 儲存 CSV
csv_path = os.path.join(os.path.dirname(video_path), "coordinates.csv")
hand_headers = [f"R{i}_{a}" for i in range(21) for a in ("x", "y", "z")] +                [f"L{i}_{a}" for i in range(21) for a in ("x", "y", "z")]
face_headers = [f"F{i}_{a}" for i in range(68) for a in ("x", "y", "z")]
header = ["frame"] + hand_headers + face_headers

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(output_data)

messagebox.showinfo("完成", f"已儲存：\n{csv_path}")
