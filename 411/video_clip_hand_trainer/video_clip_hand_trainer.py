import cv2
import mediapipe as mp
import numpy as np
import json
import os

VIDEO_FILE = "input_video.mp4"
GESTURES_FILE = "gestures.json"

# 初始化
if os.path.exists(GESTURES_FILE):
    with open(GESTURES_FILE, "r", encoding="utf-8") as f:
        gestures = json.load(f)
else:
    gestures = {}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_code(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = [lm.y for lm in hand.landmark[:6]]
        center = np.mean(landmarks[:5])
        return "".join(['1' if y < center else '0' for y in landmarks])
    return None

def run_clip_trainer():
    cap = cv2.VideoCapture(VIDEO_FILE)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    start_frame = None
    end_frame = None
    frame_list = []

    print("🎬 按 S 開始剪片區間，E 結束，Q 離開，空白鍵播放下一幀")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        show_frame = frame.copy()
        cv2.putText(show_frame, f"Frame: {current_frame}/{total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("剪片手勢訓練工具", show_frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            start_frame = current_frame
            print(f"▶️ 起點設定為第 {start_frame} 幀")
        elif key == ord('e'):
            end_frame = current_frame
            print(f"⏹️ 結束點設定為第 {end_frame} 幀")
            if start_frame is not None and end_frame is not None and end_frame > start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                codes = []
                for f in range(start_frame, end_frame + 1):
                    ret, fframe = cap.read()
                    if not ret:
                        break
                    code = extract_code(fframe)
                    if code:
                        codes.append(code)
                if codes:
                    from collections import Counter
                    most_common = Counter(codes).most_common(1)[0][0]
                    word = input("📝 請輸入這段片段對應的詞語：")
                    if word:
                        gestures[most_common] = word
                        print(f"✅ 已儲存手勢：{word} -> {most_common}")
            else:
                print("⚠️ 起點與終點無效")
        elif key == ord(' '):
            current_frame += 1
        elif key == ord('q'):
            break
        else:
            current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    with open(GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(gestures, f, indent=4, ensure_ascii=False)
    print("✅ 手勢資料已儲存完畢！")

if __name__ == "__main__":
    run_clip_trainer()
