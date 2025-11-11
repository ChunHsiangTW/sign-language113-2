# inference2.py
import os
import json
import torch
from Sign_Transformer_Encoder import SignTransformerEncoder  # ← 新類別名

MODEL_PATH = "./checkpoints/sign_model.pt"
LABEL_MAP_PATH = "./label_map.json"
DATA_FOLDER = "./json_data"
DO_NORM = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# label_map
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# input_dim
some_file = next(fn for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json"))
with open(os.path.join(DATA_FOLDER, some_file), "r", encoding="utf-8") as f:
    sample0 = json.load(f)
input_dim = len(sample0["frames"][0]["keypoints"])  # 應為 330

# model（需與 train.py 超參一致）
model = SignTransformerEncoder(
    input_size=input_dim,
    d_model=256,
    num_layers=3,
    nhead=4,
    dim_feedforward=512,
    num_classes=num_classes,
    dropout=0.2
).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

def preprocess_frames_varlen(frames, do_norm=False):
    """
    回傳：x:[1,T,330]（不 pad）、lengths:[1]
    """
    if not frames:
        return None, None
    x = torch.tensor([fr["keypoints"] for fr in frames], dtype=torch.float32)  # [T,330]
    if do_norm:
        mean = x.mean(dim=0, keepdim=True)
        std  = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
    T = x.size(0)
    x = x.unsqueeze(0).to(DEVICE)  # [1,T,330]
    lengths = torch.tensor([T], dtype=torch.long, device=DEVICE)
    return x, lengths

if __name__ == "__main__":
    correct = total = 0

    for fn in os.listdir(DATA_FOLDER):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(DATA_FOLDER, fn)
        with open(path, "r", encoding="utf-8") as f:
            sample = json.load(f)

        x, lengths = preprocess_frames_varlen(sample.get("frames", []), do_norm=DO_NORM)
        if x is None:
            print(f"{fn} | 無有效資料，跳過")
            continue

        gt = sample.get("label", "未知")
        with torch.no_grad():
            logits, _ = model(x, lengths=lengths)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = idx_to_label[pred_idx]

        print(f"{fn} | Ground Truth: {gt} | Prediction: {pred_label}")

        if gt in label_map:
            total += 1
            correct += int(pred_label == gt)

    acc = correct / total if total else 0.0
    print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {acc:.4f}")
