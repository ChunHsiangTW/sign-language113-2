import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 建立 label map（建議你未來改成從一個 JSON 或 Excel 檔案讀入）
def load_label_map(label_path):
    if label_path.endswith(".json"):
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError("目前只支援 JSON 格式的 label map")
    
LABEL_MAP = load_label_map("label_map.json")

class SignLanguageDataset(Dataset):
    def __init__(self, json_folder, label_map_path):
        self.samples = []
        for fname in os.listdir(json_folder):
            if fname.endswith(".json"):
                fpath = os.path.join(json_folder, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                keypoints = data["frames"]  # list of frames, each frame is a list of keypoints
                label = data["label"]

                if label not in LABEL_MAP:
                    raise ValueError(f"Label '{label}' not found in LABEL_MAP")

                # Tensor shape: (sequence_len, input_dim)
                frame_vectors = [frame["keypoints"] for frame in keypoints]
                keypoints_tensor = torch.tensor(frame_vectors, dtype=torch.float32)
                label_tensor = torch.tensor(LABEL_MAP[label], dtype=torch.long)

                self.samples.append((keypoints_tensor, label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def pad_collate_fn(batch):
    """自定義 collate_fn，會自動補齊不同長度的序列"""
    sequences, labels = zip(*batch)
    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)

    padded_sequences = [
        F.pad(seq, (0, 0, 0, max_len - seq.size(0)))  # 對時間維度補 0
        for seq in sequences
    ]
    padded_sequences = torch.stack(padded_sequences)
    labels = torch.stack(labels)
    return padded_sequences, labels

def get_dataloader(json_folder, label_map_path, batch_size=8, shuffle=True, drop_last=False):
    dataset = SignLanguageDataset(json_folder, label_map_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=pad_collate_fn  #  關鍵
    )
    print(f"訓練樣本數：{len(dataset)}")
    return loader

# 測試範例：
if __name__ == "__main__":
    folder_path = "./json_data"  # 你的 JSON 資料夾
    loader = get_dataloader(folder_path, batch_size=2)

    for batch_x, batch_y in loader:
        print("Batch input shape:", batch_x.shape)  # (batch, seq_len, input_dim)
        print("Batch labels:", batch_y)
        break



