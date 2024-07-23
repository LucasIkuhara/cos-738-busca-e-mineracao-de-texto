## Resultados sem Stemmer

| Métrica           | Resultado     |
| ----------------- | ------------- |
| Mean F1           | 0.0           |
| Mean Precision@5  | 0.04          |
| Mean Precision@10 | 0.04          |
| Mean R-Precision  | Não completou |
| MAP               | -1.04         |
| MRR               | 0.04          |
| NDCG              | 0.04          |
![[prec-11pnts-nostemmer.png]]

![[r-prec-nostemmer.png]]

## Resultados com Stemmer

O stemmer foi adicionado no arquivo de processamento de consulta e pode ser habilitado usando a variável STEMMER=True

Sua implementação corresponde ao seguinte trecho:
```python
    if STEMMER:
        stemmer = PorterStemmer()
        df.QueryText = df.QueryText.apply(lambda s: stemmer.stem(s))
```

Aqui estão os resultados com o stemmer
![[prec-11pnts-stemmer.png]]
![[r-prec-stemmer.png]]

| Métrica           | Resultado     |
| ----------------- | ------------- |
| Mean F1           | 0.0           |
| Mean Precision@5  | 0.04          |
| Mean Precision@10 | 0.04          |
| Mean R-Precision  | Não completou |
| MAP               | -1.04         |
| MRR               | 0.04          |
| NDCG              | 0.04          |

## Comparação

Avaliando os resultados, pode-se perceber que o modelo foi ineficiente na resolução do problema, mas não conseguiu-se descobrir a causa. Provavelmente, ocorreu algum erro em sua implementação, mas não é claro em que etapa isto ocorre.

Por este motivo, os resultados com e sem stemmer foram indistinguíveis