import torch
import torch.nn as nn
import numpy as np

# 讀取文字檔並建立字典
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for idx, ch in enumerate(chars)}

vocab_size = len(chars)

# 調整模型結構和訓練時一樣
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)      # embedding_dim = 256
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)  # 2層LSTM
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

model = CharLSTM(vocab_size)
model.load_state_dict(torch.load('model.pt'))
model.eval()

def generate_text(start_str, length=1000):
    model.eval()
    input_seq = torch.tensor([char2idx[ch] for ch in start_str], dtype=torch.long).unsqueeze(0)
    hidden = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))  # num_layers=2，batch=1，hidden_size=256
    generated = start_str

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        last_char_logits = output[0, -1]
        p = torch.nn.functional.softmax(last_char_logits, dim=0).detach().cpu().numpy()
        next_idx = np.random.choice(len(chars), p=p)
        next_char = idx2char[next_idx]
        generated += next_char
        input_seq = torch.tensor([[next_idx]], dtype=torch.long)

    return generated

# 範例：從「從前從前」開始生成1000字
output = generate_text("從前從前", length=1000)

with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(output)

print("生成文章已儲存到 generated_text.txt")
