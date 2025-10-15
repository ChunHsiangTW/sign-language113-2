import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

DATA_NPZ = "data/dataset_3d.npz"
MODEL_OUT = "models/transformer_3d.keras"

EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4

data = np.load(DATA_NPZ)
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
num_classes = len(np.unique(np.concatenate([y_train, y_test])))

seq_len, feat_dim = X_train.shape[1], X_train.shape[2]
print(f"✅ X_train={X_train.shape}, X_test={X_test.shape}, classes={num_classes}")

class PositionalEncoding(layers.Layer):
    def __init__(self, length, dim):
        super().__init__()
        pos = np.arange(length)[:, None]
        i = np.arange(dim)[None, :]
        angle = pos / np.power(10000, (2 * (i // 2)) / dim)
        pe = np.zeros_like(angle)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

def transformer_block(x, heads=4, dim_ff=256, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=64)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(dim_ff, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization()(x + ff)

inp = layers.Input(shape=(seq_len, feat_dim))
x = PositionalEncoding(seq_len, feat_dim)(inp)
x = transformer_block(x)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inp, out)
model.compile(optimizer=optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True)], verbose=1)

os.makedirs("models", exist_ok=True)
model.save(MODEL_OUT)
print(f"✅ 模型已儲存：{MODEL_OUT}")
