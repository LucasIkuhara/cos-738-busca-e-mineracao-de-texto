# %%
from utils import read_cfg, setup_logging, tokenize_sequence
import logging
from lxml import etree
import pandas as pd


setup_logging("Gerador de Lista Invertida")
cfg = read_cfg("GLI.CFG")
logging.debug("Arquivo de configuração lido com sucesso")

# %%
def read_docs(doc_files: list[str]) -> pd.DataFrame:

    logging.info(f"Iniciando leitura de {len(doc_files)} arquivos")

    QUERY_SCHEMA = "cfc-2.dtd"

    # Read schema
    with open(f"data/{QUERY_SCHEMA}") as f:
        dtd = etree.DTD(f)

    df_data = []
    read = 0

    for doc_file in doc_files:
        # Read query
        with open(f"data/{doc_file}") as f:
            xml = etree.parse(f)

        # Validate schema
        if not dtd.validate(xml):
            logging.error(f"Arquivo {doc_file} lido falhou na checagem de schema")
            continue

        root = xml.getroot()

        for record in root.findall('RECORD'):
            record_num = record.find('RECORDNUM').text.strip()

            # Check for abstract
            if (abs:= record.find('ABSTRACT')) is not None:
                df_data += [(record_num, abs.text.strip())]

            elif (ext := record.find('EXTRACT')) is not None:
                df_data += [(record_num, ext.text.strip())]

        read += 1
        logging.info(f"Arquivo {doc_file} lido com sucesso")

    df = pd.DataFrame(df_data, columns=['ID', 'TEXT'])
    logging.info(f"Lidos {read} arquivos com sucesso. O dataframe gerado tem {len(df)} linhas")

    return df

# %%

def transform_word_freq(df_raw: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando tranformação de lista inversa")
    
    data = {}
    for doc, id in zip(df_raw.iloc, range(len(df_raw))):

        txt = doc["TEXT"]
        sentence = tokenize_sequence(txt)
        for word in sentence:

            if not data.get(word):
                data[word] = [0 for _ in range(len(df_raw))]

            data[word][id] += 1

    data = [(word, data[word]) for word in data]
    df = pd.DataFrame(data, columns=("WORD", "FREQUENCY"))
    print(df.head())

    logging.info("Transformação de lista inversa finalizado com sucesso")
    return df

df = read_docs(cfg["LEIA"])
reverse_list = transform_word_freq(df)

# %%
# Get file name
reverse_list_file = cfg["ESCREVE"].pop()
reverse_list.to_csv(
    reverse_list_file,
    sep=";",
    index=False
)
