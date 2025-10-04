# sign_lstm_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    標準正弦位置編碼（不需學習），形狀：[B, T, d_model]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)           # [T, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)   # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0)                         # [1, T, d_model]
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class SignTransformerEncoder(nn.Module):
    """
    序列分類的 Transformer Encoder：
    - Input: x [B,T,330], lengths [B]
    - Output: logits [B,C], aux(None for now)
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
        self.proj = nn.Linear(input_size, d_model)   # 330 -> d_model
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,        # 讓輸入維持 [B,T,C]
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model, num_classes)   # 分類頭（masked mean pooling 後）

    @staticmethod
    def _make_key_padding_mask(lengths: torch.Tensor, T_max: int) -> torch.Tensor:
        """
        由 lengths 產生 key_padding_mask：形狀 [B, T_max]，True 代表「要遮」的 padding。
        """
        device = lengths.device
        idx = torch.arange(T_max, device=device).unsqueeze(0)        # [1, T_max]
        mask = idx >= lengths.unsqueeze(1)                           # [B, T_max]
        return mask  # True=pad

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D], mask: [B, T] (True=pad)
        回傳：忽略 padding 的平均池化 [B, D]
        """
        mask_f = (~mask).float().unsqueeze(-1)   # 有效位置=1.0
        x = x * mask_f
        denom = mask_f.sum(dim=1).clamp_min(1e-6)
        return x.sum(dim=1) / denom

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        x: [B, T, 330]; lengths: [B]（有效長度，不含 padding）
        回傳：logits [B,C], aux(None)
        """
        z = self.proj(x)           # [B,T,d_model]
        z = self.pos(z)            # 加位置訊息

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._make_key_padding_mask(lengths, z.size(1))  # [B,T]

        # Transformer Encoder（每層都會用 key_padding_mask 忽略補齊）
        z = self.encoder(z, src_key_padding_mask=key_padding_mask)   # [B,T,d_model]
        z = self.norm(z)

        # 池化：masked mean pooling（比 CLS token 更穩一些）
        if key_padding_mask is None:
            pooled = z.mean(dim=1)
        else:
            pooled = self._masked_mean(z, key_padding_mask)          # [B,d_model]

        logits = self.cls(self.dropout(pooled))                       # [B,C]
        return logits, None
