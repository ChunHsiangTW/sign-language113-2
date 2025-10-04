# prepare_data.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SignLanguageDataset(Dataset):
    """
    讀每支 JSON，使用 frames[*]['keypoints'] -> [T, 330]（可變長度）
    回傳：x:[T,330]（此處不 pad）、y:int、length:int (=T)
    """
    def __init__(self, json_folder: str, label_map_path: str, normalize: bool = False):
        super().__init__()
        self.json_folder = json_folder
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

        if self.normalize:
            mean = x.mean(dim=0, keepdim=True)
            std  = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        T = x.size(0)
        label_name = data.get("label", None)
        if label_name is None:
            raise ValueError(f"{path} missing 'label'")
        if label_name not in self.label_map:
            raise ValueError(f"Label '{label_name}' not in label_map.json")
        y = self.label_map[label_name]

        return x, y, T

def collate_fn(batch):
    """
    batch: list of (x:[T_i,330], y:int, len_i:int)
    回傳：
      xs     : [B, T_max, 330]（時間維用 0 padding）
      ys     : [B]
      lengths: [B]（原始 T_i）
    """
    xs_list, ys_list, lens_list = zip(*batch)
    xs = pad_sequence(xs_list, batch_first=True, padding_value=0.0)  # [B, T_max, 330]
    ys = torch.tensor(ys_list, dtype=torch.long)                     # [B]
    lengths = torch.tensor(lens_list, dtype=torch.long)              # [B]
    return xs, ys, lengths

def get_dataloader(folder_path: str,
                   label_map_path: str,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   normalize: bool = False,
                   num_workers: int = 0,
                   pin_memory: bool = False) -> DataLoader:
    dataset = SignLanguageDataset(folder_path, label_map_path, normalize=normalize)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)

if __name__ == "__main__":
    dl = get_dataloader("./json_data", "./label_map.json", batch_size=2, normalize=False)
    for X, y, L in dl:
        print("X:", X.shape, "y:", y.shape, "lengths:", L)
        break
