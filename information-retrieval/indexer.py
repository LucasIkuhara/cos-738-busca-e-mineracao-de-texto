# %%
import pandas as pd
from utils import read_cfg, setup_logging
import numpy as np
import logging


setup_logging("Indexador")
cfg = read_cfg("INDEX.CFG")
reverse_freq_file = cfg["LEIA"].pop()
model_file = cfg["ESCREVA"].pop()
logging.debug("Arquivo de configuração lido com sucesso")

# %%
# Reverse list reading
logging.info(f"Iniciando leitura de lista invertida")

freq_by_doc = pd.read_csv(reverse_freq_file, sep=";")
freq_by_doc.FREQUENCY = freq_by_doc.FREQUENCY.apply(
    lambda ls: np.fromstring(ls[1:-1], sep=",")
)

logging.info(f"Finalizada a leitura de lista invertida")


# %%
# Index filtering
index = freq_by_doc
st_size = len(index)
logging.info(f"Filtrando palavras do índice. Iniciando com {st_size}")

# Only words with 2 or more letters
index = index[index.WORD.str.len() > 1]
logging.info(f"Removidas palavras com menos de 2 letras. Restam {len(index)} palavras.")

# Only letters, no numbers
index = index[index.WORD.str.isalpha()]
logging.info(f"Removidas palavras com números. Restam {len(index)} palavras.")

# %%
# TD/IDF
logging.info(f"Aplicando TF/IDF")
term_freq_acc = index.FREQUENCY.sum()
term_amount = len(index)

# Frequency of term T in document D / Total terms in document D
index["TF"] = index.FREQUENCY.apply(lambda word_freq: word_freq / term_freq_acc)
# index["IDF"] = index.FREQUENCY / term_amount

# %%
# Write model
index.to_csv(
    model_file,
    index=False
)
logging.info(f"Modelo salvo no arquivo {model_file} com sucesso")
