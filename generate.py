import torch
import torch.nn as nn
import numpy as np
import random

# 載入字典與模型
char_to_idx, idx_to_char = torch.load('dictionary.pt')
vocab_size = len(char_to_idx)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

model = CharLSTM(vocab_size)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# 起頭句清單，請確保字元均存在字典中
start_phrases = ["從前從前", "有一個人", "故事開始於", "某個夜晚"]

# 篩選有效起頭句
valid_start_phrases = [phrase for phrase in start_phrases if all(ch in char_to_idx for ch in phrase)]
if not valid_start_phrases:
    raise ValueError("起頭句中所有字元不在字典裡，請檢查字典與起頭句。")

start_str = random.choice(valid_start_phrases)

def generate_text(start_str, length=2000, temperature=0.8):
    input_seq = torch.tensor([char_to_idx[ch] for ch in start_str], dtype=torch.long).unsqueeze(0)
    hidden = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))
    generated = start_str

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        logits = output[0, -1] / temperature
        probs = torch.nn.functional.softmax(logits, dim=0).detach().cpu().numpy()
        next_idx = np.random.choice(vocab_size, p=probs)
        next_char = idx_to_char[next_idx]
        generated += next_char
        input_seq = torch.tensor([[next_idx]], dtype=torch.long)

    return generated.replace('<EOS>', '') + "\n\n--- 文章結束 ---"

output = generate_text(start_str)

# 儲存到檔案
with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(output)

print("生成文章已儲存到 generated_text.txt")
print("文章開頭:", start_str)
print("生成內容預覽：")
print(output[:1000])  # 印出前1000字預覽
