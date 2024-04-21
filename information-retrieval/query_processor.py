# %%
from read_cfg import read_cfg
import logging
from lxml import etree
import pandas as pd


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename=f'{__file__}.log', level=logging.INFO, format=FORMAT)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Iniciando módulo de Processamento de Consulta")


cfg = read_cfg("PC.CFG")
logging.debug("Arquivo de configuração lido com sucesso")


# %%
def read_query(query_file: str) -> pd.DataFrame:

    logging.info(f"Iniciando leitura do arquivo {query_file}")

    class ValidationException(Exception):
        pass

    # Read query
    with open(f"data/{query_file}") as f:
        xml = etree.parse(f)

    # Read schema
    with open("data/cfc-2.dtd") as f:
        dtd = etree.DTD(f)

    if dtd.validate(xml):
        df = pd.read_xml(etree.tostring(xml))

    else:
        logging.error(f"Arquivo {query_file} lido falhou na checagem de schema")
        raise ValidationException

    logging.info(f"Arquivo {query_file} lido com sucesso")

    return df
# %%
read_query("cf74.xml")
# %%
