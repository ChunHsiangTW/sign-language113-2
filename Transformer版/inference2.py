# inference2.py － 支援用 train.py 輸出的名單一鍵評估，不必切資料夾
import os, json, warnings
import torch
from torch.utils.data import Subset, DataLoader

# 關鍵：直接建 Dataset/Subset，而不是用 get_dataloader（因為要篩選檔案）
from prepare_data import SignLanguageDataset, collate_fn
from Sign_Transformer_Encoder import SignTransformerEncoder

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage*")

# ===== 基本設定 =====
MODEL_PATH      = "./checkpoints/sign_model.pt"
LABEL_MAP_PATH  = "./label_map.json"
DATA_FOLDER     = "./json_data"      # 仍用同一個資料夾，不必手動切
SPLIT_DIR       = "./splits"         # train.py 輸出的名單放這
TRAIN_LIST_PATH = os.path.join(SPLIT_DIR, "train_files.txt")
VAL_LIST_PATH   = os.path.join(SPLIT_DIR, "val_files.txt")

EVAL_SPLIT = "val"   # "val" 或 "train"
BATCH_SIZE = 8
DO_NORM    = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 載入標籤表 =====
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}
num_classes = len(label_map)

# ===== 偵測 input_dim =====
some_file = next((fn for fn in os.listdir(DATA_FOLDER) if fn.endswith(".json")), None)
if some_file is None:
    raise FileNotFoundError(f"No .json files under {DATA_FOLDER}")
with open(os.path.join(DATA_FOLDER, some_file), "r", encoding="utf-8") as f:
    input_dim = len(json.load(f)["frames"][0]["keypoints"])

# ===== 建模（需與 train.py 一致）=====
model = SignTransformerEncoder(
    input_size=input_dim, d_model=256, num_layers=3, nhead=4,
    dim_feedforward=512, num_classes=num_classes, dropout=0.2
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== 建 Dataset；若有名單就做 Subset =====
dataset = SignLanguageDataset(
    json_folder=DATA_FOLDER,
    label_map_path=LABEL_MAP_PATH,
    normalize=DO_NORM
)

def read_list(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return None

want_files = None
used_split = "train"

if EVAL_SPLIT.lower() == "val":
    s = read_list(VAL_LIST_PATH)
    if s: 
        want_files = s
        used_split = "val"
elif EVAL_SPLIT.lower() == "train":
    s = read_list(TRAIN_LIST_PATH)
    if s:
        want_files = s
        used_split = "train"

if want_files is not None:
    # 將 dataset.files 依名單篩成索引清單
    idxs = [i for i, fp in enumerate(dataset.files) if fp in want_files]
    if not idxs:
        print(f"[warn] {used_split} 名單存在，但在 {DATA_FOLDER} 找不到對應檔案；將評估整個資料夾。")
        subset = dataset
    else:
        subset = Subset(dataset, idxs)
else:
    # 找不到名單檔 → 退回整個資料夾
    subset = dataset
    used_split = "all" if EVAL_SPLIT.lower() not in ("train", "val") else EVAL_SPLIT.lower()
    print(f"[info] 找不到 {EVAL_SPLIT} 名單檔，將評估整個資料夾。")

loader = DataLoader(
    subset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn
)

# 逐檔列印：用 subset 版本的 files 來對齊
# 注意：Subset 沒有 files 屬性，所以我們取自 dataset.files，再照 idxs 切一份
if isinstance(subset, Subset):
    base_files = dataset.files
    idxs = subset.indices
    file_list = [base_files[i] for i in idxs]
else:
    file_list = dataset.files

print(f"[Split: {used_split}] Evaluating {len(file_list)} samples from {DATA_FOLDER}")

total = correct = 0
ptr = 0

with torch.no_grad():
    for X, y, L in loader:
        X, y, L = X.to(DEVICE), y.to(DEVICE), L.to(DEVICE)
        logits, _ = model(X, lengths=L)
        preds = logits.argmax(dim=1).cpu()

        B = y.size(0)
        batch_files = file_list[ptr:ptr + B]
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
print(f"\n[Split: {used_split}] Eval on {total} samples  |  Accuracy: {acc:.4f}")
