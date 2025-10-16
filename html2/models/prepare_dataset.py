#製作資料3

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("aligned_keypoints.csv")

X = data.drop(columns=["filename", "frame", "label"]).values
y = data["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
np.save("label_classes.npy", encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
np.savez("dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("✅ 已建立 dataset.npz 訓練資料")
