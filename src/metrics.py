import numpy as np


def hits_per_user(predicted, labels, k):
    if labels is None:
        return None

    predicted = np.array(predicted[:k]).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1) # в исходнике было 1, -1

    return (predicted == labels).sum(1)


def recall_per_user(predicted, labels, k):
    hits = hits_per_user(predicted, labels, k)

    if hits is None:
        return 0.0

    labels_len = min(k, len(labels))
    return hits.sum() / labels_len


def precision_per_user(predicted, labels, k):
    hits = hits_per_user(predicted, labels, k)

    if hits is None:
        return 0.0

    return hits.sum() / k


def ndcg_per_user(predicted, labels, k):
    hits = hits_per_user(predicted, labels, k)

    pred_len = min(k, len(predicted))
    labels_len = min(k, len(labels))

    weights = 1.0 / np.log2(np.arange(k) + 2)

    dcg = (hits * weights[:pred_len]).sum()
    idcg = sum(weights[:labels_len])

    return dcg / idcg


# def ndcg(recommendations, ground_truth, k):
#     groupped_recommendations = recommendations[['user_id', 'item_id']].groupby("user_id").apply(
#             lambda row: list(row)["item_id"],
#             include_groups=False,
#         )
    
#     groupped_ground_truth = ground_truth[['user_id', 'item_id']].groupby("user_id").apply(
#             lambda row: list(row)["item_id"],
#             include_groups=False,
#         )

#     ndcg = 


    
    