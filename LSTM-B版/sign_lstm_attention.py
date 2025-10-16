# sign_lstm_attention.py  —— 直接整檔覆蓋這份即可
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output, mask=None):
        """
        lstm_output: [B, T, H]
        mask: [B, T]，True 代表「要遮（無效/補齊）」；False 代表有效
        """
        # 打分
        scores = self.attn(lstm_output).squeeze(-1)  # [B, T]

        # 遮住補齊的位置，避免被分到注意力
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # 注意力權重
        alpha = F.softmax(scores, dim=1)             # [B, T]
        # 加權求和
        context = torch.bmm(alpha.unsqueeze(1), lstm_output).squeeze(1)  # [B, H]
        return context, alpha  # alpha 回傳 [B, T]（更好看）

class SignLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)   # 仍維持單向，與你訓練一致
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        """
        x: [B, T, D]
        lengths: [B]，每筆序列的有效幀數（不含補零）
        """
        lstm_out, _ = self.lstm(x)                   # [B, T, H]

        # 建 mask：True=要遮（無效/pad）
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            B, T, _ = lstm_out.size()
            time_idx = torch.arange(T, device=lstm_out.device).unsqueeze(0).expand(B, T)
            mask = time_idx >= lengths.unsqueeze(1)   # True=要遮
        else:
            lstm_out, _ = self.lstm(x)
            mask = None

        context, alpha = self.attention(lstm_out, mask=mask)
        logits = self.fc(context)
        return logits, alpha
