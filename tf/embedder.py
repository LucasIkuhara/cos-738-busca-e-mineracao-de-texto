# %%
# Imports
import ollama
import pandas as pd
import numpy as np
import pickle

# %%
# Read data
FILE = "test_raw.csv"
OUT_FILE = f"emb_{FILE.split("_")[0]}.parquet"

df = pd.read_csv(FILE)

# %%
# Perform embeddings
def embed(st: str):
    try:
        emb = ollama.embeddings(model="llama3", prompt=st)
        return emb["embedding"]

    except Exception as err:
        print(f"Failed to encode line: \n\n{st}")
        print(err)
        return "<failed>"

df["embedding"] = df["statement"].apply(embed)

# %%
# Export results as parquet
df.to_parquet(OUT_FILE, index=False)

# %%
# Export as pickles for faster loading
# Vector series to 2d np array
x = np.stack(df["embedding"].map(np.array))

# String labels to sparse floats
def y_to_one_hot(y_df: pd.dfFrame) -> np.array:
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
        y_df[cl] = y_df["status"] == cl
        y_df[cl] = y_df[cl].astype(np.float32)

    y_df = y_df.drop(columns=["status"])
    return y_df.to_numpy()

y_df = df[["status"]]
y = y_to_one_hot(y_df)

# Dump data as pickle files
pickle.dump(x, open("/mnt/d/bmt-data/x.pickle", "wb"), protocol=None, fix_imports=True, buffer_callback=None)
pickle.dump(y, open("/mnt/d/bmt-data/y.pickle", "wb"), protocol=None, fix_imports=True, buffer_callback=None)
