# %%
from utils import read_cfg, setup_logging
import logging
import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
from unidecode import unidecode


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

    df = pd.DataFrame(df_data, columns=['RECORDNUM', 'ABSTRACT'])
    logging.info(f"Lidos {read} arquivos com sucesso. O dataframe gerado tem {len(df)} linhas")

    return df

# %%

def transform_expected_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando tranformação de resultados esperados")
    df = df_raw[["QueryNumber", "Item", "ItemScore"]]
    df.columns = ["QueryNumber","DocNumber", "DocVotes"]

    logging.info("Tranformação de resultados esperados finalizado com sucesso")
    return df

# %%
df = read_docs(cfg["LEIA"])

# Get file name
expected_results_file = cfg["ESPERADOS"].pop()

# %%
queries = transform_queries(df)
queries.to_csv(
    queries_file,
    sep=";",
    index=False
)
