# %%
from read_cfg import read_cfg
import logging
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

    if dtd.validate(xml):
        df = pd.read_xml(etree.tostring(xml))

    else:
        logging.error(f"Arquivo {query_file} lido falhou na checagem de schema")
        raise ValidationException

    logging.info(f"Arquivo {query_file} lido com sucesso")

    return df

# %%
def transform_queries(df_raw: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando tranformação de consultas")
    df = df_raw[["QueryNumber", "QueryText"]]

    df.QueryText = df.QueryText.str.replace(";", "")
    df.QueryText = df.QueryText.str.upper()
    df.QueryText = df.QueryText.apply(lambda s: unidecode(s))

    logging.info("Tranformação de consultas finalizado com sucesso")
    return df

def transform_expected_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando tranformação de resultados esperados")
    df = df_raw[["QueryNumber", "DocNumber", "DocVotes"]]

    logging.info("Tranformação de resultados esperados finalizado com sucesso")
    return df

# %%

df = read_query("cfquery.xml")

# Get file names
queries_file = cfg["CONSULTAS"].pop()
expected_results_file = cfg["ESPERADOS"].pop()

# %%
transform_queries(df).to_csv(
    queries_file,
    sep=";",
    index=False
)

transform_expected_results(df).to_csv(
    expected_results_file,
    sep=";",
    index=False
)

# %%
