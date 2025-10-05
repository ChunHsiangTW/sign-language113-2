# train.py ─ 分層 80/20 + 存 train/val 名單 + 同時列印 train_acc/val_acc + 以 val_acc 存最佳
import os, json, random, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Subset, DataLoader
from collections import defaultdict

from prepare_data import SignLanguageDataset, collate_fn
from Sign_Transformer_Encoder import SignTransformerEncoder

# ===== 超參數 =====
DATA_FOLDER = "./json_data"
LABEL_MAP_PATH = "./label_map.json"
CHECKPOINT_DIR = "./checkpoints"
LATEST_PATH   = os.path.join(CHECKPOINT_DIR, "sign_model.pt")  # 最新 / 給 inference2.py 讀
SPLIT_DIR = "./splits"  # 方案A：名單輸出位置

BATCH_SIZE = 8
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 0.01
NORMALIZE = False  # 若改 True，inference2.py 也要改成 True

# Transformer 組態（務必與 inference2.py 一致）
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 512
DROPOUT = 0.2

# ===== 固定種子（可重現）=====
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

# 偵測 input_dim
some_file = next((os.path.join(DATA_FOLDER, fn) for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json")), None)
if some_file is None:
    raise FileNotFoundError(f"No .json files under {DATA_FOLDER}")
with open(some_file, "r", encoding="utf-8") as f:
    sample0 = json.load(f)
input_dim = len(sample0["frames"][0]["keypoints"])  # 應為 330

# ===== 建立完整 Dataset（可變長度，batch 端補零）=====
dataset = SignLanguageDataset(
    json_folder=DATA_FOLDER,
    label_map_path=LABEL_MAP_PATH,
    normalize=NORMALIZE
)
print(f"總樣本數：{len(dataset)}")

# ===== 分層 80/20 切分（每類各自 8:2；可重現）=====
indices_per_cls = defaultdict(list)
for idx, path in enumerate(dataset.files):
    with open(path, "r", encoding="utf-8") as f:
        lab_name = json.load(f)["label"]
    y = dataset.label_map[lab_name]  # 0..C-1
    indices_per_cls[y].append(idx)

train_idx, val_idx = [], []
rng = random.Random(42)
for cls, idx_list in indices_per_cls.items():
    rng.shuffle(idx_list)
    k = int(round(len(idx_list) * 0.8))
    train_idx.extend(idx_list[:k])
    val_idx.extend(idx_list[k:])

print(f"Train: {len(train_idx)}  |  Val: {len(val_idx)}")

# ===== 方案A：輸出名單（不搬檔案，inference 直接讀名單）=====
os.makedirs(SPLIT_DIR, exist_ok=True)
with open(os.path.join(SPLIT_DIR, "train_files.txt"), "w", encoding="utf-8") as f:
    for i in train_idx:
        f.write(dataset.files[i] + "\n")
with open(os.path.join(SPLIT_DIR, "val_files.txt"), "w", encoding="utf-8") as f:
    for i in val_idx:
        f.write(dataset.files[i] + "\n")
print(f"已輸出名單到 {SPLIT_DIR}/train_files.txt 與 {SPLIT_DIR}/val_files.txt")

# ===== Subset 與 DataLoader =====
train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn
)

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

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_val_acc = -1.0

@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for X, y, L in loader:
        X, y, L = X.to(device), y.to(device), L.to(device)
        logits, _ = model(X, lengths=L)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return (correct / total) if total else 0.0

for epoch in range(1, EPOCHS + 1):
    # === Train 一個 epoch ===
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y, lengths in train_loader:
        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
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
    train_acc  = total_correct / total_samples
    scheduler.step()

    # === Eval on Val（關閉 Dropout）===
    val_acc = evaluate_on_loader(model, val_loader, device)

    # === 以 val_acc 存最佳模型；檔名帶 epoch/分數 + 複製一份 latest ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(CHECKPOINT_DIR, f"sign_model_e{epoch:02d}_val{val_acc:.4f}.pt")
        torch.save(model.state_dict(), save_path)
        shutil.copyfile(save_path, LATEST_PATH)

    print(
        f"Epoch [{epoch}/{EPOCHS}]  "
        f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
        f"val_acc={val_acc:.4f}  best_val_acc={best_val_acc:.4f}"
    )

print(f"最佳模型（latest）已儲存至：{LATEST_PATH}")
