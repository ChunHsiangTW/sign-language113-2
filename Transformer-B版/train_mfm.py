# train_mfm.py
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from prepare_data import get_dataloader
from Sign_Transformer_Encoder import SignTransformerEncoder

# ===== 超參數 =====
DATA_FOLDER = "./json_data"
LABEL_MAP_PATH = "./label_map.json"
CKPT_OUT = "./checkpoints/pretrain_mfm.pt"

BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.2
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 512
NORMALIZE = False
P_MASK = 0.15      # 遮蔽比例（時間幀）
NOISE_STD = 0.05   # 10% 隨機噪聲強度

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_time_mask(lengths: torch.Tensor, p_mask: float) -> torch.Tensor:
    """
    依每筆有效長度，產生要遮蔽的幀布林遮罩 [B,T_max]（True=要遮）
    保證每條序列至少有 1 個被遮幀。
    """
    B, T_max = lengths.size(0), int(lengths.max().item())
    idx = torch.arange(T_max, device=lengths.device).unsqueeze(0).expand(B, T_max)  # [B,T]
    valid = idx < lengths.unsqueeze(1)  # True=有效
    # 對「有效位置」用伯努力抽樣
    prob = torch.full((B, T_max), p_mask, device=lengths.device)
    bern = torch.bernoulli(prob).bool()
    mask = bern & valid
    # 至少 1 幀
    empty = (mask.sum(dim=1) == 0)
    if empty.any():
        for b in torch.nonzero(empty, as_tuple=False).flatten():
            t_last = int(lengths[b].item()) - 1
            if t_last >= 0:
                mask[b, t_last] = True
    return mask  # True=要遮蔽（計算 MSE 的位置）

def apply_bert_style_replacement(x: torch.Tensor, mask: torch.Tensor, noise_std: float):
    """
    x: [B,T,330], mask: [B,T] True=要替換
    BERT 80/10/10：80% -> 0、10% -> 噪聲、10% -> 保留原值
    回傳 x_masked
    """
    B, T, D = x.shape
    device = x.device
    x_masked = x.clone()

    # 把 [B,T] 攤平成一維索引方便分配
    where = mask.nonzero(as_tuple=False)  # [K,2] (b,t)
    K = where.size(0)
    if K == 0:
        return x_masked  # 沒遮就原樣

    # 隨機打亂後切 80/10/10
    order = torch.randperm(K, device=device)
    k80 = int(0.8 * K)
    k90 = int(0.9 * K)
    idx80 = where[order[:k80]]      # 0
    idx10n = where[order[k80:k90]]  # noise
    # idx10p = where[order[k90:]]   # keep (不用動)

    if k80 > 0:
        x_masked[idx80[:,0], idx80[:,1], :] = 0.0

    if (k90 - k80) > 0:
        noise = torch.randn((k90 - k80, D), device=device) * noise_std
        x_masked[idx10n[:,0], idx10n[:,1], :] = noise

    return x_masked

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 讀 input_dim / num_classes
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    some_file = next(fn for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json"))
    with open(os.path.join(DATA_FOLDER, some_file), "r", encoding="utf-8") as f:
        sample0 = json.load(f)
    input_dim = len(sample0["frames"][0]["keypoints"])  # 應為 330

    loader = get_dataloader(
        folder_path=DATA_FOLDER,
        label_map_path=LABEL_MAP_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        normalize=NORMALIZE,
        pin_memory=torch.cuda.is_available()
    )
    print(f"MFM 預訓練樣本數：{len(loader.dataset)}")

    model = SignTransformerEncoder(
        input_size=input_dim,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        dim_feedforward=DIM_FF,
        num_classes=len(label_map),
        dropout=DROPOUT
    ).to(device)

    optimz = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimz, T_max=EPOCHS)
    mse = nn.MSELoss(reduction="none")  # 之後只在被遮位置平均

    best_loss = float("inf")
    os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, denom = 0.0, 0

        for x, _, lengths in loader:            # y 不用
            x = x.to(device)                    # [B,T_max,330]
            lengths = lengths.to(device)        # [B]

            # 產生時間幀遮罩（僅有效區）
            time_mask = make_time_mask(lengths, P_MASK)  # True=要遮
            time_mask = time_mask.to(device)

            # BERT 風格替換（80/10/10）
            x_masked = apply_bert_style_replacement(x, time_mask, NOISE_STD)

            # 前向：encode -> reconstruct
            z = model.encode(x_masked, lengths)            # [B,T_max,d_model]
            recon = model.reconstruct_head(z)              # [B,T_max,330]

            # 只在「被遮的幀」計 MSE
            diff = mse(recon, x)                           # [B,T_max,330]
            frame_mask = time_mask.unsqueeze(-1).float()   # [B,T_max,1]
            masked_loss = (diff * frame_mask).sum() / frame_mask.sum().clamp_min(1.0)

            optimz.zero_grad()
            masked_loss.backward()
            optimz.step()

            total += masked_loss.item() * x.size(0)
            denom += x.size(0)

        avg = total / max(denom, 1)
        scheduler.step()

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), CKPT_OUT)

        print(f"[MFM] Epoch {epoch:02d}/{EPOCHS}  loss={avg:.6f}  best={best_loss:.6f}")

    print(f"預訓練完成，已存：{CKPT_OUT}")

if __name__ == "__main__":
    main()
