# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score


# %%
resultados = pd.read_csv('resultados/RESULTADOS.csv', delimiter=";")
resultados_esperados = pd.read_csv('resultados/RESULTADOS_ESPERADOS.csv', delimiter=";")

print(resultados.head())
print(resultados_esperados.head())

# %%
# Calcular precisão e recall
## Processar os resultados para obter listas de resultados para cada consulta
def parse_result_string(result_string):
    # Remove os colchetes e divide a string em partes
    result_string = result_string.strip('[]')
    items = result_string.split(', ')
    
    # Agrupa os itens em tuplas de três elementos
    result_list = [(int(items[i]), int(items[i+1]), float(items[i+2])) for i in range(0, len(items), 3)]
    return result_list

resultados['Result'] = resultados['Result'].apply(parse_result_string)
resultados_dict = resultados.groupby('QueryNumber')['Result'].apply(list).to_dict()

# Processar os resultados esperados para obter listas de documentos relevantes para cada consulta
resultados_esperados_dict = resultados_esperados.groupby('QueryNumber')['DocNumber'].apply(list).to_dict()

# Exemplo de visualização dos dados processados
print(resultados_dict)
print(resultados_esperados_dict)

# Função para calcular a curva de precisão e recall em 11 pontos
def precision_recall_curve_points(resultados_dict, resultados_esperados_dict, num_points=11):
    all_recall_levels = []
    all_precision_at_recall = []

    for query_number in resultados_dict:
        if query_number not in resultados_esperados_dict:
            continue
        
        result_list = resultados_dict[query_number][0]  # Obter a lista de resultados
        expected_list = resultados_esperados_dict[query_number]

        # Criar y_true e y_scores
        y_true = [1 if doc[1] in expected_list else 0 for doc in result_list]
        y_scores = [doc[2] for doc in result_list]

        # Calcular a curva de precisão e recall
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        recall_levels = np.linspace(0, 1, num_points)
        precision_at_recall = []

        for recall_level in recall_levels:
            precisions = precision[recall >= recall_level]
            if len(precisions) == 0:
                precision_at_recall.append(0.0)
            else:
                precision_at_recall.append(max(precisions))

        all_recall_levels.append(recall_levels)
        all_precision_at_recall.append(precision_at_recall)
    
    # Calcular a média dos valores de precisão em diferentes níveis de recall
    mean_recall_levels = np.mean(all_recall_levels, axis=0)
    mean_precision_at_recall = np.mean(all_precision_at_recall, axis=0)

    return mean_recall_levels, mean_precision_at_recall

def precision_at_k(y_true, y_scores, k):
    order = np.argsort(y_scores)[::-1]
    y_true_at_k = np.array(y_true)[order][:k]
    return np.mean(y_true_at_k)

def r_precision(y_true, y_scores):
    num_relevant = np.sum(y_true)
    order = np.argsort(y_scores)[::-1]
    y_true_at_r = np.array(y_true)[order][:num_relevant]
    return np.mean(y_true_at_r)

def average_precision(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return np.sum(precision * (recall[1:] - recall[:-1]))

def reciprocal_rank(y_true, y_scores):
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = np.array(y_true)[order]
    for i, val in enumerate(y_true_sorted):
        if val:
            return 1 / (i + 1)
    return 0

def dcg(y_true, y_scores, k):
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = np.array(y_true)[order][:k]
    return np.sum((2**y_true_sorted - 1) / np.log2(np.arange(1, k + 1) + 1))

def ndcg(y_true, y_scores, k):
    best_dcg = dcg(sorted(y_true, reverse=True), y_scores, k)
    actual_dcg = dcg(y_true, y_scores, k)
    return actual_dcg / best_dcg if best_dcg != 0 else 0

# Calcular todas as métricas para todas as queries
all_f1_scores = []
all_precision_at_5 = []
all_precision_at_10 = []
all_r_precision = []
all_map = []
all_mrr = []
all_ndcg = []

for query_number in resultados_dict:
    if query_number not in resultados_esperados_dict:
        continue

    result_list = resultados_dict[query_number][0]
    expected_list = resultados_esperados_dict[query_number]

    y_true = [1 if doc[1] in expected_list else 0 for doc in result_list]
    y_scores = [doc[2] for doc in result_list]
    y_pred = [1 if score >= 0.5 else 0 for score in y_scores]  # Exemplo de threshold

    all_f1_scores.append(f1_score(y_true, y_pred))
    all_precision_at_5.append(precision_at_k(y_true, y_scores, 5))
    all_precision_at_10.append(precision_at_k(y_true, y_scores, 10))
    all_r_precision.append(r_precision(y_true, y_scores))
    all_map.append(average_precision(y_true, y_scores))
    all_mrr.append(reciprocal_rank(y_true, y_scores))
    all_ndcg.append(ndcg(y_true, y_scores, len(y_true)))

mean_f1 = np.mean(all_f1_scores)
mean_precision_at_5 = np.mean(all_precision_at_5)
mean_precision_at_10 = np.mean(all_precision_at_10)
mean_r_precision = np.mean(all_r_precision)
mean_map = np.mean(all_map)
mean_mrr = np.mean(all_mrr)
mean_ndcg = np.mean(all_ndcg)

print('Mean F1:', mean_f1)
print('Mean Precision@5:', mean_precision_at_5)
print('Mean Precision@10:', mean_precision_at_10)
print('Mean R-Precision:', mean_r_precision)
print('MAP:', mean_map)
print('MRR:', mean_mrr)
print('NDCG:', mean_ndcg)

# Gráfico de 11 pontos de precisão e recall
recall_levels, precision_at_recall = precision_recall_curve_points(resultados_dict, resultados_esperados_dict)
plt.plot(recall_levels, precision_at_recall, marker='o')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (11 points) for All Queries')
plt.show()

# Histograma de R-Precision (comparativo)
plt.hist(all_r_precision, bins=20, alpha=0.7, label='R-Precision')
plt.xlabel('R-Precision')
plt.ylabel('Frequency')
plt.title('Histogram of R-Precision')
plt.legend()
plt.show()

