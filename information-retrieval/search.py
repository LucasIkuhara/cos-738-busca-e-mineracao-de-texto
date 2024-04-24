# %%
from pickle import load
from utils import read_cfg, setup_logging, tokenize_sequence
import logging
import numpy as np
import pandas as pd


setup_logging("Busca")
cfg = read_cfg("BUSCA.CFG")
model_file = cfg["MODELO"].pop()
results_file = cfg["RESULTADOS"].pop()

logging.debug("Arquivo de configuração lido com sucesso")

# %%
# Define model
class RetrievalModel:

    def __init__(self, weights_df: pd.DataFrame):
        self.data = weights_df

    def __call__(self, *args, **kwds):
        return self.inference(*args, **kwds)

    def inference(self, x: str) -> np.array:
        word_vecs = []
        doc = tokenize_sequence(x)

        for word in doc:
            word_vec = self.data.TFIDF[self.data.WORD == word]

            # Discard words not found
            if len(word_vec) < 1:
                continue
        
            word_vecs.append(word_vec)

        # Compute sum of all word vectors
        x= word_vecs[0]
        print(len(x))
        print(type(x))
        doc_vec = np.array(word_vecs).sum()

        return x


# Read weights and instance model

weights = pd.read_csv(model_file)
model = RetrievalModel(weights)
x = model.inference("Hello from the plains")
# %%
