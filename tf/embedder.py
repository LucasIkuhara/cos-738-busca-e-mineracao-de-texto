# %%
# Imports
import ollama
import pandas as pd

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
# Export results
df.to_parquet(OUT_FILE, index=False)
