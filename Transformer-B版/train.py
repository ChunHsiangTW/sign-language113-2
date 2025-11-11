# train.py
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import get_dataloader
from Sign_Transformer_Encoder import SignTransformerEncoder  # ← 新類別名

# ===== 超參數 =====
DATA_FOLDER = "./json_data"
LABEL_MAP_PATH = "./label_map.json"
CHECKPOINT_PATH = "./checkpoints/sign_model.pt"
PRETRAIN_PATH = "./checkpoints/pretrain_mfm.pt"

BATCH_SIZE = 8
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.01
NORMALIZE = False  # 若改 True，推論端也要 True

# Transformer 組態
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 512
DROPOUT = 0.2

# ===== 固定種子 =====
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 讀 label_map / input_dim =====
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
num_classes = len(label_map)

some_file = next(fn for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json"))
with open(os.path.join(DATA_FOLDER, some_file), "r", encoding="utf-8") as f:
    sample0 = json.load(f)
input_dim = len(sample0["frames"][0]["keypoints"])  # 應為 330

# ===== DataLoader（可變長度；batch 端 pad）=====
train_loader = get_dataloader(
    folder_path=DATA_FOLDER,
    label_map_path=LABEL_MAP_PATH,
    batch_size=BATCH_SIZE,
    shuffle=True,
    normalize=NORMALIZE,
    pin_memory=torch.cuda.is_available()
)
print(f"訓練樣本數：{len(train_loader.dataset)}")

# ===== Model / Loss / Optim / Scheduler =====
model = SignTransformerEncoder(
    input_size=input_dim,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    nhead=NHEAD,
    dim_feedforward=DIM_FF,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

if os.path.exists(PRETRAIN_PATH):
    print(f"載入預訓練權重：{PRETRAIN_PATH}")
    state = torch.load(PRETRAIN_PATH, map_location=device)
    model.load_state_dict(state, strict=False)  # 分類頭不存在沒關係

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
best_acc = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y, lengths in train_loader:
        batch_x = batch_x.to(device)     # [B,T_max,330]
        batch_y = batch_y.to(device)     # [B]
        lengths = lengths.to(device)     # [B]

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
    scheduler.step()

    # 先用訓練集挑 best；若你有驗證集，請改用驗證指標挑存檔
    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Epoch [{epoch}/{EPOCHS}]  loss={train_loss:.4f}  acc={train_acc:.4f}  best_acc={best_acc:.4f}")

print(f"最佳模型已儲存至：{CHECKPOINT_PATH}")


