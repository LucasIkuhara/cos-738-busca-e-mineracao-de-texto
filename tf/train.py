# %%
# Imports
from sklearn.model_selection import cross_val_score, train_test_split
from keras.api.models import Sequential
from scikeras.wrappers import KerasClassifier
from keras.api.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from typing import Callable
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
def make_model_sized(
        n: int,
        act_fn: str
    ) -> Callable[[], keras.Model]:

    def make_model() -> keras.Model:
        model = Sequential([
            Input(shape=(4096,)),
            Dense(n, activation=act_fn),
            Dropout(0.5),
            Dense(n, activation=act_fn),
            Dropout(0.5),
            Dense(7, activation="softmax")
        ])

        model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model
    
    make_model.__name__ = f"make_model_{n}"
    return make_model

# %%
# Train and evaluate model with cross validation
MAX_EPOCHS=100
BATCH_SIZE = 200

early_stop = EarlyStopping(monitor='loss', patience=3)

for size in [1024, 2048, 4096]:
    factory = make_model_sized(size, "relu")
    estimator = KerasClassifier(
        build_fn=factory, 
        epochs=MAX_EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=[early_stop],
        verbose=0
    )
    sc = cross_val_score(estimator, x, y, cv=5)

    print(f"[{size}-model] Accuracy Score: Mean={sc.mean():.3f} Std={sc.std():.3f}")

# %%
