# %%
from pickle import load
from utils import read_cfg, setup_logging, tokenize_sequence
import logging
import numpy as np
import pandas as pd
from time import time


start = time()
setup_logging("Busca")
cfg = read_cfg("BUSCA.CFG")
model_file = cfg["MODELO"].pop()
queries_file = cfg["CONSULTAS"].pop()
results_file = cfg["RESULTADOS"].pop()

logging.debug("Arquivo de configuração lido com sucesso")

# %%
# Define model
class RetrievalModel:

    def __init__(self, weights_df: pd.DataFrame):
        self.data = weights_df
        self._compute_doc_vecs()

    def __call__(self, *args, **kwds):
        return self.inference(*args, **kwds)
    
    def _compute_doc_vecs(self):
        '''
        Get a document vector for every article we trained on.
        '''
        vecs = self.data.TFIDF
        vecs = np.array(vecs.to_list())
        self.vecs = vecs.T

    def text_to_vec(self, txt: list[str]):
        '''
        Compute the vector for the new input
        '''
        in_vec = pd.DataFrame(self.data.WORD)
        in_vec["NEW_VEC"] = np.zeros(len(in_vec), dtype=int)

        for word in txt:
            in_vec.loc[in_vec.WORD == word, "NEW_VEC"] += 1
        
        in_vec.NEW_VEC = in_vec.NEW_VEC.apply(lambda n: n/len(txt))
        in_vec.NEW_VEC = in_vec.NEW_VEC * self.data.IDF

        return in_vec.NEW_VEC.to_numpy(dtype=np.float64)

    def cosine_similarity(self, x: np.array, y: np.array) -> float:
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def inference(self, x: str):
        '''
        Compute the distance from every article
        '''

        doc = tokenize_sequence(x)
        in_vec = self.text_to_vec(doc)
        cos_vec = [self.cosine_similarity(in_vec, vec) for vec in self.vecs]

        return cos_vec

# Read weights and instance model
logging.debug("Lendo pesos do modelo")
with open(model_file, "rb") as fd:
    weights = load(fd)
logging.debug("Pesos do modelo lidos com sucesso")

model = RetrievalModel(weights)
x = model.inference("Hello THe from the plains")

# %%
# Read queries
queries = pd.read_csv(queries_file, sep=";")

data = []
for query in queries.iloc:
    res = model.inference(query.QueryText)
    res = pd.DataFrame({"score": res, "doc_index": list(range(len(res)))})
    res = res.sort_values("score", ascending=False)

    for vec, pos in zip(res.iloc, range(1, len(res) + 1)):
        data.append([query.QueryNumber, [pos, int(vec.doc_index), vec.score]])

results = pd.DataFrame(data, columns=["QueryNumber", "Result"])

# %%
# Salvar resultados
results.to_csv(
    results_file,
    index=False,
    sep=";"
)
logging.info(f"Arquivo {results_file} criado.")
logging.info(f"Execução do modulo finalizada em {time() - start}s")
