# sign_lstm_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output, mask=None):
        """
        lstm_output: [B, T_max, H]
        mask: [B, T_max]，True=要遮（padding）
        """
        scores = self.attn(lstm_output).squeeze(-1)  # [B, T_max]
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        alpha = F.softmax(scores, dim=1)             # [B, T_max]
        context = torch.bmm(alpha.unsqueeze(1), lstm_output).squeeze(1)  # [B, H]
        return context, alpha

class SignLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        """
        x: [B, T_max?, D]（可變長度，可能已在 batch 端 pad 到 T_max）
        lengths: [B]（每筆序列原始長度，不含 padding）
        """
        if lengths is not None:
            # 只把有效時間步送進 LSTM
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, T_max, H]
            B, T_max, _ = lstm_out.size()
            time_idx = torch.arange(T_max, device=lstm_out.device).unsqueeze(0).expand(B, T_max)
            mask = time_idx >= lengths.unsqueeze(1)  # True=pad
        else:
            lstm_out, _ = self.lstm(x)
            mask = None

        context, alpha = self.attention(lstm_out, mask=mask)
        logits = self.fc(context)
        return logits, alpha
