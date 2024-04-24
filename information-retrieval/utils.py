import logging
from unidecode import unidecode
import nltk


nltk.download('punkt')

def read_cfg(filename: str) -> dict[str, list[str]]:
    with open(filename, "r") as f:

        cfg = {}

        cmds = f.readlines()
        for raw_line in cmds:
            command, arg = raw_line.split("=")
            arg = arg.replace("\n", "")

            if command in cfg:
                cfg[command].append(arg)

            else:
                cfg[command] = [arg]
        
        return cfg


def setup_logging(module_name: str):
    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'resultados/{module_name}.log', level=logging.DEBUG, format=FORMAT)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Iniciando módulo: {module_name}")


def tokenize_sequence(txt: str) -> list[str]:
    txt = unidecode(txt.upper())
    txt = txt. replace(";", "")
    sentence = nltk.tokenize.word_tokenize(txt)

    return sentence

class ValidationException(Exception):
    """Falha na validação de schema do arquivo XML. O formato difere do DTD."""
    pass