# %%
# Imports
from sklearn.model_selection import cross_val_score, train_test_split
from keras.api.models import Sequential
from scikeras.wrappers import KerasClassifier
from keras.api.layers import Dense, Input
from keras.callbacks import EarlyStopping
import keras
import pandas as pd
import pickle
import numpy as np

# %%
# Read train data
print("Dataset read started.")
x = pickle.load(open("/mnt/d/bmt-data/x.pickle", "rb"))
y = pickle.load(open("/mnt/d/bmt-data/y.pickle", "rb"))
print("Dataset read successfully.")

# %%
# Model creation
def make_model():
    model = Sequential([
        # Input(shape=(4096,)),
        Dense(1024, activation="relu", input_dim=4096),
        Dense(1024, activation="relu"),
        Dense(7, activation="softmax")
    ])

    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

# %%
# Train model
MAX_EPOCHS=100
early_stop = EarlyStopping(monitor='loss', patience=3)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=0
# )
x_train = x[:42144]
x_test = x[42144:]

y_train = y[:42144]
y_test = y[42144:]

model = make_model()
model.fit(x_train, y_train, batch_size=200, epochs=MAX_EPOCHS, callbacks=[early_stop])

# %%

model.save("/mnt/d/bmt-data/model.keras")

# %%
# Evaluate metrics
if not model:
    model = keras.models.load_model("/mnt/d/bmt-data/model.keras")

estimator = KerasClassifier(model=model, epochs=MAX_EPOCHS, batch_size=200, callbacks=[early_stop])
score = estimator.score(x_test, y_test)
print("Score: ", score)

# %%
