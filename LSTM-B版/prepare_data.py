import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def resample_to_T(x: torch.Tensor, fixed_T: int) -> torch.Tensor:
    """
    將 [T, D] 以等間距取樣成 [fixed_T, D]，保留整段節奏不切掉頭尾。
    """
    T = x.size(0)
    if T == fixed_T:
        return x
    idx = torch.linspace(0, T - 1, steps=fixed_T).round().long()
    return x[idx]

class SignLanguageDataset(Dataset):
    """
    讀每支 JSON，使用 frames[*]['keypoints'] -> [T, 330]
    - 若 T >= fixed_T：等間距取樣到 fixed_T
    - 若 T <  fixed_T：尾端補 0 到 fixed_T
    - 回傳有效長度 eff_len = min(T, fixed_T)
    """
    def __init__(self, json_folder: str, label_map_path: str,
                 fixed_T: int = 140, normalize: bool = False):
        super().__init__()
        self.json_folder = json_folder
        self.fixed_T = fixed_T
        self.normalize = normalize

        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

        self.files = [
            os.path.join(json_folder, fn)
            for fn in sorted(os.listdir(json_folder))
            if fn.endswith(".json")
        ]
        if not self.files:
            raise FileNotFoundError(f"No .json found in {json_folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        if not frames:
            raise ValueError(f"{path} has no frames")

        x = torch.tensor([fr["keypoints"] for fr in frames], dtype=torch.float32)  # [T, 330]
        T = x.size(0)

        if T >= self.fixed_T:
            x = resample_to_T(x, self.fixed_T)
            eff_len = self.fixed_T
        else:
            pad = self.fixed_T - T
            x = F.pad(x, (0, 0, 0, pad))
            eff_len = T

        if self.normalize:
            mean = x.mean(dim=0, keepdim=True)
            std  = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        label_name = data.get("label", None)
        if label_name is None:
            raise ValueError(f"{path} missing 'label'")
        if label_name not in self.label_map:
            raise ValueError(f"Label '{label_name}' not in label_map.json")

        y = self.label_map[label_name]
        return x, y, eff_len  # 回傳 lengths

def collate_fn(batch):
    xs   = torch.stack([b[0] for b in batch], dim=0)                  # [B, T, 330]
    ys   = torch.tensor([b[1] for b in batch], dtype=torch.long)      # [B]
    lens = torch.tensor([b[2] for b in batch], dtype=torch.long)      # [B]
    return xs, ys, lens

def get_dataloader(folder_path: str,
                   label_map_path: str,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   fixed_T: int = 140,
                   normalize: bool = False,
                   num_workers: int = 0,
                   pin_memory: bool = False) -> DataLoader:
    dataset = SignLanguageDataset(folder_path, label_map_path, fixed_T, normalize)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)

if __name__ == "__main__":
    dl = get_dataloader("./json_data", "./label_map.json",
                        batch_size=2, fixed_T=140, normalize=False)
    for X, y, L in dl:
        print("X:", X.shape, "y:", y.shape, "lengths:", L)
        break
