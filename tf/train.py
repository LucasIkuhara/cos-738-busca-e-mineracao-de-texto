# %%
# Imports
from tensorflow import keras
from keras.api.models import Sequential
# from scikeras.wrappers import KerasClassifier
from keras.api.layers import Dense, Input
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

# %%
# Read train data
train = pd.read_parquet("/mnt/d/bmt-data/emb_train.parquet")

# Vector series to 2d np array
train["embedding"] = train["embedding"].map(eval)
x_train = np.stack(train["embedding"].map(np.array))

# String labels to sparse floats
def y_to_one_hot(y_df: pd.DataFrame) -> np.array:
    classes = [
        "Anxiety",
        "Bipolar",
        "Depression",
        "Normal",
        "Personality disorder",
        "Stress",
        "Suicidal"
    ]

    for cl in classes:
        y_df[cl] = train["status"] == cl
        y_df[cl] = y_df[cl].astype(np.float32)

    y_df = y_df.drop(columns=["status"])
    return y_df.to_numpy()

y_df = train[["status"]]
y_train = y_to_one_hot(y_df)

# %%
# Model creation
model = Sequential([
    Input(shape=(4096,)),
    Dense(4096, activation="relu"),
    Dense(4096, activation="relu"),
    Dense(7, activation="softmax")
])

model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# %%
# Train model
early_stop = EarlyStopping(monitor='loss', patience=3)
MAX_EPOCHS=1000

model.fit(x_train, y_train, batch_size=200, epochs=MAX_EPOCHS, callbacks=[early_stop])
model.save("/mnt/d/bmt-data/model.keras")

# %%
