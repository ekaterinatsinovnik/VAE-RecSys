import os
# import json
from typing import Sequence, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from clearml import Logger
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

from src.models import MultiVAE


class MultiVAETrainer:
    def __init__(
        self,
        item_num: int,
        latent_dim: int = 200,
        num_hidden: int = 1,
        hidden_dim: int = 600,
        dropout: float = 0.5,
        epochs_num: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        k: int = 10,
        beta: Union[float, int, None] = None,
        anneal_cap: float = 1.0,
        total_anneal_steps: int = 20000, #20000
        set_lr_scheduler = False,
        decay_step: int = 25,
        gamma: float = 1.0
    ):
        """_summary_

        Args:
            item_num (int): _description_
            latent_dim (int, optional): _description_. Defaults to 200.
            num_hidden (int, optional): _description_. Defaults to 1.
            hidden_dim (int, optional): _description_. Defaults to 600.
            dropout (float, optional): _description_. Defaults to 0.1.
            epochs_num (int, optional): _description_. Defaults to 10.
            learning_rate (float, optional): _description_. Defaults to 2e-3.
            weight_decay (float, optional): _description_. Defaults to 3e-6.
            k (int, optional): _description_. Defaults to 10.

            beta (Union[float, int, None], optional): If beta is a number, it's stable during training.
                                If beta is None, it's annealing from 0 to anneal_cap.
                                It's recommended to anneal beta for better performance. Defaults to None.
            anneal_cap (float, optional): Upper limit of increasing beta (best beta founded).
                                If best beta isn't founded, anneal_cap defaults to 1.0.
            total_anneal_steps (int, optional): steps number when beta riches anneal_cap. Defaults to 2000.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # или перенести создание модели в методы: типа для трейна так, для теста проверка на наличие модели после трейна, потом

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.model = MultiVAE(item_num, latent_dim, num_hidden, hidden_dim, dropout).to(
            self.device
        )

        self.learning_rate = learning_rate
        self.epochs_num = epochs_num
        self.weight_decay = weight_decay

        # self.l2_reg = l2_reg
        # self.factor = factor
        # self.patience = patience

        # функция гет оптимизер?
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # self.set_lr_scheduler = set_lr_scheduler
        # if self.set_lr_scheduler:
        #     self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=gamma)

        self.k = k

        self.anneal_amount = 1.0 / total_anneal_steps
        if isinstance(beta, (float, int)):
            self.beta = beta
            self.annealing_beta = False
        elif beta is None:
            self.beta = 0.0
            self.annealing_beta = True
            # self.current_best_metric = 0.0
            self.anneal_cap = anneal_cap
        else:
            pass
            # add raising error if beta is a number and annealing_beta is True
            # или просто дефолтом поставить а здесь проверку на тип? или проверку на тип не здесь, а в мейне?

    @property
    def _calculate_beta(self):
        if self.model.training and self.annealing_beta:
            self.beta = min(self.beta + self.anneal_amount, self.anneal_cap)
        return self.beta

    def _calculate_loss(self, output, input, mu, logvar):
        softmax_output = F.log_softmax(output, dim=1)
        CE = -torch.mean(torch.sum(softmax_output * input, dim=1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return CE + self._calculate_beta * KLD

    def _train_step(self, user_batch):

        batch = torch.FloatTensor(self.sparse_train_input[user_batch[0]].toarray()).to(self.device)

        self.optimizer.zero_grad()

        output, mu, logvar = self.model.forward(batch)
        loss = self._calculate_loss(output, batch, mu, logvar)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def _validation_step(self, user_batch, calculate_loss=False):
        input = torch.FloatTensor(self.sparse_val_input[user_batch[0]].toarray()).to(self.device)
        label = torch.FloatTensor(self.sparse_val_label[user_batch[0]].toarray()).to(self.device)

        output, mu, logvar = self.model(input)

        output[input != 0] = -float("inf") # IMPORTANT: remove items that were in the input
        flatten_output = output.flatten()
        flatten_label = label.flatten()
        user_index = (
            torch.arange(output.size(0), dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, output.size(1))
            .flatten()
        ).to(self.device)

        metric = RetrievalNormalizedDCG(top_k=self.k)(
            flatten_output, flatten_label, indexes=user_index
        ).item()


        if calculate_loss:
            val_loss = self._calculate_loss(output, input, mu, logvar).item()
            return val_loss, metric

        return metric

    # https://medium.com/@a0922/introduction-to-clearml-executing-in-google-colab-5485768cf6e9
    def save_checkpoint(self, model_path="multivae.pth"):
        # ckpt_path = (
        #     cache_path
        #     / f"model_{model.info()}_optim_{optimizer_name}_acc_{accuracy:.4f}_epoch_{epoch_id}.ckpt"
        # )
        # torch.save(self.model.state_dict(), os.path.join(gettempdir(), "multivae.pt"))

        # torch.save(self.model.state_dict(), "multivae.pt")
        # torch.save(self.model.state_dict(), os.path.join(os.getcwd(), "multivae.pt"))
        torch.save(self.model.state_dict(), os.path.join("models/ml-1m/val3", model_path))

    def train(self, dataloader, train, val):
        self.best_ndcg_metric = 0.0
        self.best_ndcg_metric_for_beta = 0.0
        self.sparse_train_input = train
        self.sparse_val_input = val['input']
        self.sparse_val_label = val['label']

        for epoch in range(self.epochs_num):
            self.model.train()

            # if self.set_lr_scheduler:
            #     self.lr_scheduler.step()

            average_train_loss = 0
            for batch_id, batch in enumerate(dataloader):
                # batch = [x.to(self.device) for x in batch]
                train_loss = self._train_step(batch)
                average_train_loss += train_loss

            Logger.current_logger().report_scalar(
                "Loss",
                "Train",
                value=average_train_loss / len(dataloader),
                iteration=epoch,
            )
            print(f'epoch = {epoch} loss={average_train_loss / len(dataloader)}')

            self.model.eval()
            with torch.no_grad():
                # tqdm_dataloader = tqdm(self.val_loader)
                average_val_loss, average_ndcg_metric = 0, 0
                # custom_metrics = {'Recall@1' : 0, 'Recall@10' :0, 'NDCG@1' : 0, 'NDCG@10' : 0}
                for batch_id, batch in enumerate(dataloader):
                    # batch = [x.to(self.device) for x in batch]

                    # current_val_loss, current_ndcg_metric = self._validation_step(
                    #     batch, calculate_loss=True
                    # )
                    current_ndcg_metric = self._validation_step(batch)
                    
                    # average_val_loss += current_val_loss
                    average_ndcg_metric += current_ndcg_metric

                # val_loss = average_val_loss / len(dataloader)
                ndcg_metric = average_ndcg_metric / len(dataloader)

                if self.best_ndcg_metric < ndcg_metric:
                    self.best_ndcg_metric = ndcg_metric
                    self.save_checkpoint()
                    
                    if self.annealing_beta:
                        self.best_beta = self.beta
                        Logger.current_logger().report_scalar(
                            "Annealing beta",
                            "beta",
                            value=self.best_beta,
                            iteration=epoch,
                        )

                # Logger.current_logger().report_scalar(
                #     "Loss",
                #     "Validation",
                #     value=val_loss,
                #     iteration=epoch,
                # )


                Logger.current_logger().report_scalar(
                    "Validation metric",
                    "NDCG@10",
                    value=ndcg_metric,
                    iteration=epoch,
                )
                print(f'epoch = {epoch} m={ndcg_metric}')

            if epoch == self.epochs_num - 1:
                self.save_checkpoint("multivae_final.pth")


    def _calculate_metrics(self, user_batch):

        input = torch.FloatTensor(self.sparse_test_input[user_batch[0]].toarray()).to(self.device)
        label = torch.FloatTensor(self.sparse_test_label[user_batch[0]].toarray()).to(self.device)

        output, _, _ = self.model(input)

        output[input != 0] = -float("inf") # IMPORTANT: remove items that were in the input
        flatten_output = output.flatten().cpu()
        flatten_label = label.flatten().cpu()
        user_index = (
            torch.arange(output.size(0), dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, output.size(1))
            .flatten()
        )
        self.metrics.reset()

        return self.metrics(flatten_output, flatten_label, user_index)

    def test(self, dataloader, test, ks: Sequence[int], model_path=None):
        self.sparse_test_input = test['input']
        self.sparse_test_label = test['label']

        if model_path is None:
            model_path = os.path.join("models/ml-1m/val3", "multivae.pth")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.ks = ks
        metrics = MetricCollection(
            [MetricCollection(
                    [
                        RetrievalNormalizedDCG(top_k=k),
                        RetrievalRecall(top_k=k),
                        RetrievalPrecision(top_k=k),
                    ],
                    postfix=f"@{k}",
                )
                for k in ks
            ]
        )

        # self.metrics = metrics.to(self.device)
        self.metrics = metrics
        self.model.eval()

        with torch.no_grad():
            result_metrics = {k : 0 for k in metrics.keys(keep_base=True)}
            # avg, avg10 = 0, 0
            for batch in dataloader:
                current_metrics = self._calculate_metrics(batch)
                # avg += current_metrics['RetrievalNormalizedDCG@1']
                # avg10 += current_metrics['RetrievalNormalizedDCG@10']
                # print(current_metrics)
                result_metrics = {k: result_metrics[k] + current_metrics[k] for k in result_metrics}

            result_metrics = {k: result_metrics[k] / len(dataloader) for k in result_metrics}

            # result_metrics = self.metrics.compute()
            # print(result_metrics)
            # avg = avg / len(dataloader)
            # avg10 = avg10 / len(dataloader)
            # Logger.current_logger().report_single_value(name='RetrievalNormalizedDCG@1', value=avg)
            # Logger.current_logger().report_single_value(name='RetrievalNormalizedDCG@10', value=avg10)

            for metric, value in result_metrics.items():
                Logger.current_logger().report_single_value(name=metric, value=value)

            # print(avg, avg10)
            # print(result_metrics['RetrievalNormalizedDCG@1'], result_metrics['RetrievalNormalizedDCG@10'])
            
            # убрать тензоры
            # with open(os.path.join("models/ml-1m/val", 'test_metrics.json'), 'w') as f:
            #     json.dump(result_metrics, f, indent=6)

def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics