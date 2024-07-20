# Information Retrieval

## Dependências

O projeto necessita de uma distribuição python de versão 3.10 ou superior, e utiliza nativamente o poetry como gerenciador de pacotes. Caso o tenha o poetry instalado, as dependências podem ser instaladas (e ativadas) com o comando:

```sh
poetry install && poetry env use
```

Alternativamente, caso o poetry não esteja instalado, as dependências podem ser instaladas via pip3 com o comando:

```sh
pip3 install -r requirements.txt
```

## Execução

Para executar todos os módulos, execute o script run_all:
```sh
./run_all.sh
```

## Observações

A métrica de distância usada foi a similaridade de cosseno. Portanto, maior é mais similar.
