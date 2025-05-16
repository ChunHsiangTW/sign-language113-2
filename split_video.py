import sys
print("你現在使用的 Python 路徑是：", sys.executable)

from moviepy.editor import VideoFileClip
import os
import time

# 影片檔案名稱
video_path = "20250425-1.mp4"

# 分割時間段（你可以自己改）
time_segments = [
    (0, 2),
    (5, 10),
    (12, 55)
]

# 取得目前時間作為檔名區分用
timestamp = time.strftime("%Y%m%d_%H%M%S")

# 建立輸出資料夾
output_folder = "output_clips"
os.makedirs(output_folder, exist_ok=True)

# 讀取影片
clip = VideoFileClip(video_path)

# 分割並儲存每個片段
for i, (start, end) in enumerate(time_segments):
    subclip = clip.subclip(start, end)

    # 避免影片變形，保持原高度
    subclip = subclip.resize(height=clip.h)

    # 加上時間戳記避免覆蓋
    output_filename = f"clip_{i+1}_{timestamp}.mp4"
    output_path = os.path.join(output_folder, output_filename)

    # 顯示進度
    print(f"正在輸出 {output_filename}（時間段：{start}秒 ~ {end}秒）")

    # 儲存影片
    subclip.write_videofile(output_path, codec="libx264")

print("✅ 所有影片分割完成！輸出於資料夾：", output_folder)
