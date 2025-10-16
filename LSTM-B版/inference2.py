import os
import json
import torch
import torch.nn.functional as F
from sign_lstm_attention import SignLSTMWithAttention

# ===== 參數 =====
MODEL_PATH = "./checkpoints/sign_model.pt"
LABEL_MAP_PATH = "./label_map.json"
DATA_FOLDER = "./json_data"
FIXED_T = 140
DO_NORM = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== label_map =====
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# ===== 動態確認 input_dim（通常為 330）=====
some_file = next(fn for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json"))
with open(os.path.join(DATA_FOLDER, some_file), "r", encoding="utf-8") as f:
    sample0 = json.load(f)
input_dim = len(sample0["frames"][0]["keypoints"])

# ===== 模型（與訓練一致）=====
model = SignLSTMWithAttention(
    input_size=input_dim,   # 330
    hidden_size=256,
    num_layers=2,
    num_classes=num_classes
).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

def resample_to_T(x: torch.Tensor, fixed_T: int) -> torch.Tensor:
    """
    將 [T, D] 以等間距取樣成 [fixed_T, D]
    """
    T = x.size(0)
    if T == fixed_T:
        return x
    idx = torch.linspace(0, T - 1, steps=fixed_T).round().long()
    return x[idx]

# ===== 前處理：等間距取樣 + 回傳 lengths =====
def preprocess_frames(frames, fixed_T=140, do_norm=False):
    if not frames:
        return None, None
    x = torch.tensor([fr["keypoints"] for fr in frames], dtype=torch.float32)  # [T, D]
    T = x.size(0)

    if T >= fixed_T:
        x = resample_to_T(x, fixed_T)
        eff_len = fixed_T
    else:
        x = F.pad(x, (0, 0, 0, fixed_T - T))  # 尾端補 0
        eff_len = T

    if do_norm:
        mean = x.mean(dim=0, keepdim=True)
        std  = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std

    x = x.unsqueeze(0).to(DEVICE)                                   # [1, T, D]
    lengths = torch.tensor([eff_len], dtype=torch.long, device=DEVICE)  # [1]
    return x, lengths

# ===== 主程式 =====
if __name__ == "__main__":
    correct = total = 0

    for fn in os.listdir(DATA_FOLDER):
        if not fn.endswith(".json"):
            continue

        path = os.path.join(DATA_FOLDER, fn)
        with open(path, "r", encoding="utf-8") as f:
            sample = json.load(f)

        x, lengths = preprocess_frames(sample.get("frames", []), fixed_T=FIXED_T, do_norm=DO_NORM)
        if x is None:
            print(f"{fn} | 無有效資料，跳過")
            continue

        gt = sample.get("label", "未知")

        with torch.no_grad():
            logits, _ = model(x, lengths=lengths)   # 傳入 lengths 做 mask /（若 forward 有 pack 也一起生效）
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = idx_to_label[pred_idx]

        print(f"{fn} | Ground Truth: {gt} | Prediction: {pred_label}")

        if gt in label_map:
            total += 1
            correct += int(pred_label == gt)

    acc = correct / total if total else 0.0
    print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {acc:.4f}")
