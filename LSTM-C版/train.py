# train.py
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import get_dataloader
from sign_lstm_attention import SignLSTMWithAttention

# ===== 可調參數 =====
DATA_FOLDER = "./json_data"
LABEL_MAP_PATH = "./label_map.json"
CHECKPOINT_PATH = "./checkpoints/sign_model.pt"

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NORMALIZE = False  # 若改 True，推論端也要 True

# ===== 固定隨機種子（小數據集很重要）=====
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 讀 label_map，推得 num_classes & input_dim =====
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
num_classes = len(label_map)

# 從資料夾中抓一個檔案確認 input_dim
some_file = next(fn for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json"))
with open(os.path.join(DATA_FOLDER, some_file), "r", encoding="utf-8") as f:
    sample0 = json.load(f)
input_dim = len(sample0["frames"][0]["keypoints"])  # 應為 330

# ===== Dataloader（可變長度；batch 端自動 pad 到該批最長）=====
train_loader = get_dataloader(
    folder_path=DATA_FOLDER,
    label_map_path=LABEL_MAP_PATH,
    batch_size=BATCH_SIZE,
    shuffle=True,
    normalize=NORMALIZE,
    pin_memory=torch.cuda.is_available()
)
print(f"訓練樣本數：{len(train_loader.dataset)}")

# ===== Model / Loss / Optim =====
model = SignLSTMWithAttention(
    input_size=input_dim,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
best_acc = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y, lengths in train_loader:
        batch_x = batch_x.to(device)         # [B, T_max, D]
        batch_y = batch_y.to(device)         # [B]
        lengths = lengths.to(device)         # [B]

        logits, _ = model(batch_x, lengths=lengths)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_y.size(0)

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    # 存最佳（以訓練集準確率為準；如有驗證集，請改用驗證集）
    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Epoch [{epoch}/{EPOCHS}]  loss={train_loss:.4f}  acc={train_acc:.4f}  best_acc={best_acc:.4f}")

print(f"最佳模型已儲存至：{CHECKPOINT_PATH}")

