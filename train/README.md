
---

train/
│
├── models/
│   ├── extract_keypoints.py
│   ├── align_keypoints.py
│   ├── prepare_dataset_3d.py
│   ├── train_transformer.py
│   └── realtime_recognition.py
│
├── data/               ← 存 CSV、JSON、dataset_3d.npz
├── static/datavideos/  ← 你的影片
├── requirements.txt
└── README.md



## 安裝套件

建議建立虛擬環境：

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
