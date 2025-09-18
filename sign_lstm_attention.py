import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(attn_weights * lstm_output, dim=1)       # [batch_size, hidden_size]
        return context, attn_weights


class SignLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)                              # [batch_size, seq_len, hidden_size]
        context, attn_weights = self.attention(lstm_out)        # [batch_size, hidden_size], [batch_size, seq_len, 1]
        output = self.fc(context)                               # [batch_size, num_classes]
        return output, attn_weights
