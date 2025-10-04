# inference2.py － 列出逐檔 GT/Pred + 總結 Acc（與訓練同 DataLoader/補零/lengths）
import os
import json
import warnings
import torch

from prepare_data import get_dataloader
from Sign_Transformer_Encoder import SignTransformerEncoder

# --- 靜音 Nested Tensor 原型 API 警告（不影響正確性）---
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage*")

MODEL_PATH = "./checkpoints/sign_model.pt"   # 建議用 train.py 複製出的 latest
LABEL_MAP_PATH = "./label_map.json"
DATA_FOLDER = "./json_data"
BATCH_SIZE = 8
DO_NORM = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀 label_map
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# 偵測 input_dim
some_file = next(fn for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json"))
with open(os.path.join(DATA_FOLDER, some_file), "r", encoding="utf-8") as f:
    input_dim = len(json.load(f)["frames"][0]["keypoints"])

# 建模（要與 train.py 超參一致）
model = SignTransformerEncoder(
    input_size=input_dim, d_model=256, num_layers=3, nhead=4,
    dim_feedforward=512, num_classes=num_classes, dropout=0.2
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 與訓練一致的 DataLoader（關鍵！）
loader = get_dataloader(
    folder_path=DATA_FOLDER,
    label_map_path=LABEL_MAP_PATH,
    batch_size=BATCH_SIZE,
    shuffle=False,                 # 不打亂，才能對齊檔名順序
    normalize=DO_NORM,
    pin_memory=torch.cuda.is_available()
)

# 逐檔列印：用 dataset.files 依序對齊目前批次
# （不需改 prepare_data.py，只要 DataLoader 預設 SequentialSampler）
file_list = getattr(loader.dataset, "files", None)
if file_list is None:
    raise RuntimeError("Dataset 沒有 files 屬性，無法逐檔列印。請在 Dataset 中保留 self.files。")

total = correct = 0
ptr = 0  # 指向目前要印的檔名索引

with torch.no_grad():
    for X, y, L in loader:           # X:[B,Tmax,330], y:[B], L:[B]
        X, y, L = X.to(DEVICE), y.to(DEVICE), L.to(DEVICE)
        logits, _ = model(X, lengths=L)
        preds = logits.argmax(dim=1).cpu()

        B = y.size(0)
        batch_files = file_list[ptr:ptr + B]     # 依序對齊檔名
        ptr += B

        for i in range(B):
            fn = os.path.basename(batch_files[i])
            gt_label = idx_to_label[y[i].item()]
            pred_label = idx_to_label[preds[i].item()]
            print(f"{fn} | Ground Truth: {gt_label} | Prediction: {pred_label}")

            total += 1
            if gt_label == pred_label:
                correct += 1

acc = correct / total if total else 0.0
print(f"\nEval on {total} samples  |  Accuracy: {acc:.4f}")

