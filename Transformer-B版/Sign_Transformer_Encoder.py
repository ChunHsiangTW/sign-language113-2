
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1,T,d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])

class SignTransformerEncoder(nn.Module):
    """
    - 分類用 forward(x, lengths) 仍回傳 (logits, None)
    - 自監督預訓練用：呼叫 encode(...) 拿序列特徵，再經 reconstruct_head 重建到 330 維
    """
    def __init__(self,
                 input_size: int,
                 d_model: int,
                 num_layers: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 分類頭（微調時用）
        self.cls = nn.Linear(d_model, num_classes)

        # 自監督重建頭（預訓練時用）
        self.reconstruct_head = nn.Linear(d_model, input_size)

    @staticmethod
    def _make_key_padding_mask(lengths: torch.Tensor, T_max: int) -> torch.Tensor:
        idx = torch.arange(T_max, device=lengths.device).unsqueeze(0)  # [1,T]
        return idx >= lengths.unsqueeze(1)  # [B,T], True=pad

    @staticmethod
    def _masked_mean(x: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
        if pad_mask is None:
            return x.mean(dim=1)
        valid = (~pad_mask).float().unsqueeze(-1)  # [B,T,1]
        denom = valid.sum(dim=1).clamp_min(1e-6)
        return (x * valid).sum(dim=1) / denom

    def encode(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        供預訓練/微調共用：回傳每個時間步的編碼特徵 [B,T,d_model]
        """
        z = self.proj(x)
        z = self.pos(z)
        pad_mask = self._make_key_padding_mask(lengths, z.size(1)) if lengths is not None else None
        z = self.encoder(z, src_key_padding_mask=pad_mask)
        z = self.norm(z)
        return z

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        分類前向：回傳 (logits, None)
        """
        z = self.encode(x, lengths)                           # [B,T,d_model]
        pad_mask = self._make_key_padding_mask(lengths, z.size(1)) if lengths is not None else None
        pooled = self._masked_mean(z, pad_mask)               # [B,d_model]
        logits = self.cls(self.dropout(pooled))               # [B,C]
        return logits, None
