# %%
# Imports
import tensorflow as tf
from tensorflow import keras
from keras.api.models import Sequential
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.api.layers import Dense, Input
from keras import activations
from keras.losses import MeanSquaredError
import pandas as pd
import numpy as np

# %%
# Read train data
train = pd.read_parquet("/mnt/d/bmt-data/emb_train.parquet")

# Vector series to 2d np array
train["embedding"] = train["embedding"].map(eval)
x_train = np.stack(train["embedding"].map(np.array))

# String labels to sparse floats
classes = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal"
]

y_df = train[["status"]]
for cl in classes:
    y_df[cl] = train["status"] == cl
    y_df[cl] = train[cl].astype(np.float32)

y_df = y_df.drop(columns=["status"])
y_train = y_df.to_numpy()

# %%
# Model creation
model = Sequential([
    Input(shape=(4096,)),
    Dense(4096, activation="relu"),
    Dense(4096, activation="relu"),
    Dense(7, activation="softmax")
])

model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %%
# ReLu
MAX_EPOCHS=10
model.fit(x_train, y_train, batch_size=200, epochs=MAX_EPOCHS)

# %%
