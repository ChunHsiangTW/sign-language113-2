import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sign_lstm_attention import SignLSTMWithAttention
from prepare_data import get_dataloader
import os
import json


# 參數設置
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LABEL_MAP_PATH = "./label_map.json"
FOLDER_PATH = "./json_data"
MODEL_SAVE_PATH = "./checkpoints/sign_model.pt"
LOG_DIR = "./logs"

# 準備資料
train_loader = get_dataloader(
    folder_path=FOLDER_PATH,
    label_map_path=LABEL_MAP_PATH,
    batch_size=BATCH_SIZE,
    shuffle=True
)
print(f"訓練樣本數：{len(train_loader.dataset)}")


# 動態取得類別數
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
num_classes = len(label_map)

# 模型、損失、優化器
model = SignLSTMWithAttention(input_size=330, hidden_size=HIDDEN_SIZE, num_classes=num_classes, num_layers=NUM_LAYERS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# TensorBoard
writer = SummaryWriter(LOG_DIR)

# 訓練迴圈
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y, lengths in train_loader:
        batch_x  = batch_x.to(device)      # (B, T, D)
        batch_y  = batch_y.to(device)      # (B,)
        lengths  = lengths.to(device)      # (B,)

        # 傳入 lengths，讓注意力遮掉補齊
        outputs, _ = model(batch_x, lengths=lengths)

        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * batch_x.size(0)
        _, predicted  = torch.max(outputs, 1)
        total_correct += (predicted == batch_y).sum().item()
        total_samples += batch_y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch+1)
    writer.add_scalar("Accuracy/train", accuracy, epoch+1)

# 儲存模型
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"模型已儲存至 {MODEL_SAVE_PATH}")



