#訓練模型4
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

data = np.load("dataset.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

print(f"訓練資料形狀: {X_train.shape}, 測試資料形狀: {X_test.shape}")
print(f"訓練標籤範圍: {np.unique(y_train)}")
print(f"測試標籤: {y_train.shape}, 測試標籤: {y_test.shape}")   
num_classes = len(np.unique(y_train))
input_dim = X_train.shape[1]

# reshape: (samples, sequence_length, feature_dim)
X_train = X_train.reshape((X_train.shape[0], 200, input_dim))
X_test = X_test.reshape((X_test.shape[0], 200, input_dim))

inputs = layers.Input(shape=(200, input_dim))
x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))
model.save("transformer_model.h5")
print("✅ 模型訓練完成並儲存為 transformer_model.h5")
