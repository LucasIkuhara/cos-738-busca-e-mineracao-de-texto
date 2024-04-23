# %%
import pandas as pd
from utils import read_cfg


cfg = read_cfg("INDEX.CFG")
reverse_freq_file = cfg["LEIA"].pop()
model_file = cfg["ESCREVA"].pop()

# %%

freq_by_doc = pd.read_csv(reverse_freq_file, sep=";")
# %%
