from yt_dlp import YoutubeDL

ydl_opts = {
    'outtmpl': 'D:\saujuu/%(title)s.%(ext)s',  # 下載路徑與檔名格式
    'max_downloads': 500,
    'ignoreerrors': True
}

url = 'https://www.youtube.com/@tingangle'

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
