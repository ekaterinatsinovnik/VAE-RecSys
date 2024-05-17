import numpy as np
import pandas as pd


def hits_per_user(recommended, label, k):

    if recommended is None:
        return None

    recommended_array = np.array(recommended)[:k]
    label_array = np.array(label)

    hits = (recommended_array.reshape(-1, 1) == label_array.reshape(1, -1)).sum(axis=1)
    return hits

def recall_per_user(row, k) -> float:
    recommended = row['recommended']
    label = row['label']

    hits = hits_per_user(recommended, label, k)

    if hits is None:
        return 0.

    return hits.sum() / min(len(label), k)

def precision_per_user(row, k) -> float:
    recommended = row['recommended']
    label = row['label']

    hits = hits_per_user(recommended, label, k)

    if hits is None:
        return 0.

    return hits.sum() / k


def ndcg_per_user(row, k) -> float:
    recommended = row['recommended']
    label = row['label']

    hits = hits_per_user(recommended, label, k)

    if hits is None:
        return 0.

    recommended_len = min(len(recommended), k)
    label_len = min(len(label), k)
    
    ndcg_weights = 1. / np.log2(np.arange(2, k + 2))

    dcg = (hits * ndcg_weights[:recommended_len]).sum()
    
    idcg = ndcg_weights.cumsum()[label_len-1]

    return dcg / idcg

def compute_metrics(recommendations, ks):
    metrics = {}
    for k in ks:
        mean_ndcg = recommendations.apply(lambda row: ndcg_per_user(row, k=k), axis=1).mean()
        mean_recall = recommendations.apply(lambda row: recall_per_user(row, k=k), axis=1).mean()
        mean_precision = recommendations.apply(lambda row: precision_per_user(row, k=k), axis=1).mean()

        metrics.update({
                f"NDCG@{k}" : mean_ndcg,
                f"Recall@{k}" : mean_recall,
                f"Precision@{k}" : mean_precision
            })

    return metrics