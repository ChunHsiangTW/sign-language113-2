import torch
import torch.nn as nn

# 讀取文字檔，並在結尾加入特殊結束符號 <EOS>
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read() + '<EOS>'


# 建立字元列表
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 編碼文字
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# 定義模型
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

model = CharLSTM(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# 超參數
seq_length = 100
epochs = 50

for epoch in range(epochs):
    total_loss = 0
    count = 0
    for i in range(0, len(data) - seq_length, seq_length):
        inputs = data[i:i+seq_length].unsqueeze(0)      # (1, seq_length)
        targets = data[i+1:i+seq_length+1].unsqueeze(0)  # 目標是下一字

        outputs, _ = model(inputs)
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    print(f"Epoch {epoch+1}/{epochs}, 平均Loss: {total_loss/count:.4f}")

# 儲存模型與字典
torch.save(model.state_dict(), 'model.pt')
torch.save((char_to_idx, idx_to_char), 'dictionary.pt')
