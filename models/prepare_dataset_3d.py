import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

# === 基本設定 ===
DATA_DIR = "data"  # 存放所有 JSON 關鍵點資料的資料夾
OUTPUT_NPZ = os.path.join(DATA_DIR, "dataset_3d.npz")  # 最終輸出的資料集檔案
OUTPUT_LABELS = os.path.join(DATA_DIR, "label_map.npy")  # 對應的 label 映射表
MAX_FRAMES = 200  # 每支影片最多取多少幀
FEATURES = 21 * 3 * 2 + 68 * 3  # 每幀的特徵維度 (左右手 + 臉)

# === 掃描所有 JSON 檔案 ===
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
print(f"🔍 發現 {len(files)} 個 JSON 檔案")

X, y = [], []
label_map = {}
idx = 0  # 用來給每個 label 編號

# === 處理每一個 JSON 檔 ===
for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 🟢 原始標籤（影片對應的意思）
    # 有些 JSON 裡有 "label" 欄位，但我們更保險地根據檔名歸類
    label = obj.get("label", os.path.splitext(fname)[0])

    # 🔹 移除檔名中的數字，例如「早1」、「早2」→「早」
    label = ''.join([c for c in label if not c.isdigit()]).strip()

    # 🔹 確保 label 不為空
    if not label:
        print(f"⚠️ {fname} 沒有有效標籤，跳過")
        continue

    frames = obj["frames"]
    fixed_frames = []

    # === 防呆修正：確保每幀長度一致 ===
    for i, frame in enumerate(frames):
        flat = np.array(frame, dtype=np.float32).flatten()

        # 如果這一幀長度不符合預期，就補零或截斷
        if len(flat) != FEATURES:
            print(f"⚠️ {fname} 第 {i} 幀長度異常 ({len(flat)}≠{FEATURES}) → 自動補零修正")
            if len(flat) < FEATURES:
                flat = np.pad(flat, (0, FEATURES - len(flat)), mode="constant")
            else:
                flat = flat[:FEATURES]
        fixed_frames.append(flat.tolist())

    # === 補幀或截斷：統一長度為 MAX_FRAMES ===
    if len(fixed_frames) < MAX_FRAMES:
        pad = MAX_FRAMES - len(fixed_frames)
        fixed_frames.extend([[0.0] * FEATURES] * pad)
    else:
        fixed_frames = fixed_frames[:MAX_FRAMES]

    # === 建立 label 對應表 ===
    if label not in label_map:
        label_map[label] = idx
        idx += 1

    X.append(fixed_frames)
    y.append(label_map[label])

# === 轉成 numpy 陣列 ===
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# === 分割訓練 / 測試資料 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

# === 儲存結果 ===
os.makedirs(DATA_DIR, exist_ok=True)
np.savez(OUTPUT_NPZ, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
np.save(OUTPUT_LABELS, label_map, allow_pickle=True)

# === 結果輸出 ===
print(f"✅ Dataset 建立完成：{OUTPUT_NPZ}")
print(f"✅ Label Map（共 {len(label_map)} 類）：" )
print(label_map)
