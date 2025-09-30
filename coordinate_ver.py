import os
import csv
import cv2
import dlib
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# === Open file dialog ===
tk.Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    print("No video selected.")
    exit()

if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("❌ Missing shape_predictor_68_face_landmarks.dat")
    exit()

# === Init models ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Init video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0
coordinates_data = []
labels_data = []
start_frame = None
paused = False

# === Scroll control ===
scroll_offset = 0
line_height = 14

def mouse_scroll(event, x, y, flags, param):
    global scroll_offset
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # scroll up
            scroll_offset = max(scroll_offset - line_height, 0)
        else:          # scroll down
            scroll_offset += line_height

cv2.namedWindow("Landmark Annotator")
cv2.setMouseCallback("Landmark Annotator", mouse_scroll)

# Save last frame and coordinates
last_frame = None
face_coords = [(0.0, 0.0, 0.0)] * 68
R_hand = [(0.0, 0.0, 0.0)] * 21
L_hand = [(0.0, 0.0, 0.0)] * 21

# === Helper function: detect landmarks ===
def detect_landmarks(frame):
    global face_coords, R_hand, L_hand

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_coords = [(0.0, 0.0, 0.0)] * 68
    for face in faces:
        landmarks = predictor(gray, face)
        face_coords = [(landmarks.part(i).x, landmarks.part(i).y, 0.0) for i in range(68)]

    R_hand = [(0.0, 0.0, 0.0)] * 21
    L_hand = [(0.0, 0.0, 0.0)] * 21
    handedness = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            handedness.append(hand_info.classification[0].label)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            hand_pts = [(int(x * frame.shape[1]), int(y * frame.shape[0]), z) for x, y, z in hand_pts]
            if handedness[idx] == 'Right':
                R_hand = hand_pts
            else:
                L_hand = hand_pts
    return face_coords, L_hand, R_hand, results

# === Main loop ===
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1000, 750))  # ensure same size for detection & display
            last_frame = frame.copy()
            face_coords, L_hand, R_hand, results = detect_landmarks(frame)
        else:
            frame = last_frame.copy() if last_frame is not None else np.zeros((750,1000,3),dtype=np.uint8)

        # === Draw landmarks ===
        for i, (x, y, _) in enumerate(face_coords):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv2.putText(frame, f"F{i+1}", (int(x)+3, int(y)-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        for i, (x, y, _) in enumerate(L_hand):
            if x > 0 and y > 0:
                cv2.putText(frame, f"L{i}", (x+3, y-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        for i, (x, y, _) in enumerate(R_hand):
            if x > 0 and y > 0:
                cv2.putText(frame, f"R{i}", (x+3, y-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if not paused and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === Right panel ===
        info_width = 350
        frame = cv2.copyMakeBorder(frame, 0, 0, 0, info_width, cv2.BORDER_CONSTANT, value=(50, 50, 50))

        col1_x, col2_x = 1020, 1180
        y_offset = 30
        cv2.putText(frame, "=== Landmark Info ===", (col1_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        lines_col1 = [f"F{i+1}: ({int(x)}, {int(y)}, {z:.2f})" for i, (x, y, z) in enumerate(face_coords)]
        lines_col2 = [f"L{i}: ({x}, {y}, {z:.2f})" for i, (x, y, z) in enumerate(L_hand)]
        lines_col2 += [f"R{i}: ({x}, {y}, {z:.2f})" for i, (x, y, z) in enumerate(R_hand)]

        start_idx = scroll_offset // line_height
        visible_lines = (frame.shape[0] - 180) // line_height

        y = y_offset + 20
        for line in lines_col1[start_idx:start_idx+visible_lines]:
            cv2.putText(frame, line, (col1_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
            y += line_height
        y = y_offset + 20
        for line in lines_col2[start_idx:start_idx+visible_lines]:
            cv2.putText(frame, line, (col2_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
            y += line_height

        # === Save coordinates ===
        flat_data = [frame_idx]
        for pt in R_hand + L_hand + face_coords:
            flat_data.extend(pt)
        coordinates_data.append(flat_data)

        # === Progress bar ===
        bar_x1, bar_x2 = 10, 900
        bar_y1, bar_y2 = frame.shape[0] - 20, frame.shape[0] - 10
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 100, 100), -1)
        progress = frame_idx / max(1, total_frames)
        progress_x = int(bar_x1 + progress * (bar_x2 - bar_x1))
        cv2.rectangle(frame, (bar_x1, bar_y1), (progress_x, bar_y2), (255, 200, 0), -1)
        cv2.putText(frame, f"{progress*100:.1f}%", (bar_x2+10, bar_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # === Controls help ===
        help_y = frame.shape[0] - 80
        cv2.putText(frame, "[SPACE] Play/Pause   [A] Back 30   [D] Forward 30", (10, help_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "[Q] Back 1   [E] Forward 1   [ESC] Exit", (10, help_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "[S] Start Label   [F] End Label", (10, help_y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Landmark Annotator", frame)

        # === Keyboard control ===
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a'):
            frame_idx = max(frame_idx - 30, 0); paused = False
        elif key == ord('d'):
            frame_idx = min(frame_idx + 30, total_frames-1); paused = False
        elif key == ord('q'):
            frame_idx = max(frame_idx - 1, 0); paused = False
        elif key == ord('e'):
            frame_idx = min(frame_idx + 1, total_frames-1); paused = False
        elif key == ord('s'):
            start_frame = frame_idx
            print(f"[START] Label from frame {start_frame}")
        elif key == ord('f') and start_frame is not None:
            end_frame = frame_idx
            label = input("Enter label: ")
            labels_data.append([start_frame, end_frame, label])
            print(f"[DONE] {label}: {start_frame} ~ {end_frame}")
            start_frame = None

        if not paused:
            frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# === Save CSV files ===
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

with open("labels.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["start_frame", "end_frame", "label"])
    writer.writerows(labels_data)

print("✅ Saved coordinates.csv and labels.csv")
