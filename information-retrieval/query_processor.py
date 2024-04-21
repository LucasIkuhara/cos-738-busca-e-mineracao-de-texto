# %%
from read_cfg import read_cfg
import logging
import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
from unidecode import unidecode


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename=f'{__file__}.log', level=logging.INFO, format=FORMAT)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Iniciando módulo de Processamento de Consulta")


cfg = read_cfg("PC.CFG")
logging.debug("Arquivo de configuração lido com sucesso")


# %%
def read_query(query_file: str) -> pd.DataFrame:

    logging.info(f"Iniciando leitura do arquivo {query_file}")

    QUERY_SCHEMA = "cfcquery-2.dtd"
    class ValidationException(Exception):
        pass

    # Read query
    with open(f"data/{query_file}") as f:
        xml = etree.parse(f)

    # Read schema
    with open(f"data/{QUERY_SCHEMA}") as f:
        dtd = etree.DTD(f)

    # Validate schema
    if not dtd.validate(xml):
        logging.error(f"Arquivo {query_file} lido falhou na checagem de schema")
        raise ValidationException

    root = xml.getroot()

    data = []
    for query in root.findall('QUERY'):
        query_number = query.find('QueryNumber').text.strip()
        query_text = query.find('QueryText').text.strip()
        results = query.find('Results').text.strip()
        for record in query.find('Records'):
            item = record.text.strip()
            item_score = record.attrib['score']
            data.append([query_number, query_text, results, item, item_score])

    df = pd.DataFrame(data, columns=['QueryNumber', 'QueryText', 'Results', 'Item', 'ItemScore'])
    logging.info(f"Arquivo {query_file} lido com sucesso")

    return df


# %%
def transform_queries(df_raw: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando tranformação de consultas")
    df = df_raw[["QueryNumber", "QueryText"]]

    df.QueryText = df.QueryText.str.replace(";", "")
    df.QueryText = df.QueryText.str.upper()
    df.QueryText = df.QueryText.apply(lambda s: unidecode(s))
    df = df[df.duplicated().__invert__()]

    logging.info("Tranformação de consultas finalizado com sucesso")
    return df

def transform_expected_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando tranformação de resultados esperados")
    df = df_raw[["QueryNumber", "Item", "ItemScore"]]
    df.columns = ["QueryNumber","DocNumber", "DocVotes"]

    logging.info("Tranformação de resultados esperados finalizado com sucesso")
    return df

# %%

df = read_query("cfquery.xml")

# Get file names
queries_file = cfg["CONSULTAS"].pop()
expected_results_file = cfg["ESPERADOS"].pop()

# %%
queries = transform_queries(df)
queries.to_csv(
    queries_file,
    sep=";",
    index=False
)

expected_results = transform_expected_results(df)
expected_results.to_csv(
    expected_results_file,
    sep=";",
    index=False
)
