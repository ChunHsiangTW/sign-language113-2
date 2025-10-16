import numpy as np

# 載入標籤檔
classes = np.load("label_classes.npy", allow_pickle=True)

# 列出所有可辨識詞
print("✅ 模型可辨識的手語詞彙如下：")
for i, c in enumerate(classes):
    print(f"{i+1}. {c}")
