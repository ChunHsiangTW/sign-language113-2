# merge_csv.py
import pandas as pd
import os
import chardet  # ✅ 自動偵測編碼（若沒有要 pip install chardet）

DATA_DIR = "data"
output_file = "aligned_keypoints.csv"

all_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_keypoints.csv")]
print(f"🧩 偵測到 {len(all_files)} 個 CSV，要進行合併...\n")

dfs = []

for f in all_files:
    path = os.path.join(DATA_DIR, f)
    # --- 自動偵測編碼 ---
    with open(path, "rb") as file:
        result = chardet.detect(file.read())
        encoding = result["encoding"]

    try:
        df = pd.read_csv(path, encoding=encoding)
        # 🔧 可選：統一標籤去掉數字（例如 "不見1" → "不見"）
        df["label"] = df["label"].astype(str).str.replace(r"\d+$", "", regex=True)
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ 無法讀取檔案 {f}，錯誤：{e}")
        continue

# --- 合併所有資料 ---
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 合併完成！輸出檔案：{output_file}")
print(f"📄 合併後共有 {len(merged_df)} 筆資料")
