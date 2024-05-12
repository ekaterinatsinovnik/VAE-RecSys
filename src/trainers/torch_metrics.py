import torch


def recall(predicted, target, k):
    index = predicted.argsort(descending=True)[:, :k]
    hits = target.gather(dim=1, index=index).sum(dim=1)

    k = torch.Tensor([k]).to(target.device)
    targets_len = target.sum(dim=1).float()

    recall_per_user = hits / torch.min(k, targets_len)
    return recall_per_user.mean().cpu().item()


def ndcg(predicted, target, k):
    # scores = scores.cpu()
    # labels = labels.cpu()
    predicted_len = min(k, predicted.shape[1])
    target_len = min(k, target.shape[0])

    index = predicted.argsort(descending=True)[:, :k]
    # hits = target.gather(dim=1, index=index).sum(dim=1)
    hits = target.gather(dim=1, index=index)

    weights = 1 / torch.log2(torch.arange(2, 2 + k).float())
    print(hits)
    print(weights)
    dcg = (hits.float() * weights).sum(1)
    # idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    idcg = (target.topk(k)[0].float() * weights).sum(1)
    # idcg = torch.cumsum(weights[:k], 0)
    print(idcg)
    ndcg = dcg / idcg
    return ndcg.mean().item()
