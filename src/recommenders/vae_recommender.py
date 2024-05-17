from typing import Dict, Sequence, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from clearml import Logger
from scipy.sparse import csr_matrix
from torchmetrics import MetricCollection
from torchmetrics.retrieval import (RetrievalNormalizedDCG, 
                                    RetrievalPrecision,
                                    RetrievalRecall)

from src.models import MultiVAE


class MultiVAERecommender:
    def __init__(
        self,
        item_num: int,
        dataset: str,
        latent_dim: int = 200,
        num_hidden: int = 1,
        hidden_dim: int = 600,
        dropout: float = 0.5,
        epochs_num: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        k: int = 10,
        beta: Union[float, int, None] = None,
        anneal_cap: float = 1.0,
        total_anneal_steps: int = 20000,
        set_lr_scheduler=False,
        decay_step: int = 25,
        gamma: float = 1.0,
    ) -> None:
        """Recommender for training and testing MultVAE model.
            Possible to anneal beta coefficient in loss function.

        Args:
            item_num (int): Number of items.
            dataset (str): Dataset short name for training and testing.
            latent_dim (int, optional): Dimension of the latent representation. Defaults to 200.
            num_hidden (int, optional): Number of hidden layers in each encoder and decoder MLP layers. Defaults to 1.
            hidden_dim (int, optional): Dimension of hidden layers in encoder and decoder MLP layers. Defaults to 600.
            dropout (float, optional): Defaults to 0.1.
            epochs_num (int, optional): Defaults to 50.
            learning_rate (float, optional): Defaults to 1e-3.
            weight_decay (float, optional): Defaults to 1e-2.
            k (int, optional): Number of items for validation metric (NDCG@10). Defaults to 10.

            beta (Union[float, int, None], optional): If beta is a number, it's stable during training.
                                If beta is None, it's annealing from 0 to anneal_cap.
                                It's recommended to anneal beta for better performance. Defaults to None.
            anneal_cap (float, optional): Upper limit of increasing beta (best beta founded).
                                If best beta isn't founded, anneal_cap defaults to 1.0.
            total_anneal_steps (int, optional): steps number when beta riches anneal_cap. Defaults to 20000.
            set_lr_scheduler (bool, optional): Flag for enabling learning rate scheduler. Defaults to False.
            decay_step (int, optional): Learning rate scheduler param. Defaults to 25.
            gamma (float, optional): Learning rate scheduler param. Defaults to 1.0.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

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

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.set_lr_scheduler = set_lr_scheduler
        if self.set_lr_scheduler:
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=decay_step, gamma=gamma
            )

        self.k = k

        self.anneal_amount = 1.0 / total_anneal_steps
        if isinstance(beta, (float, int)):
            self.beta = beta
            self.annealing_beta = False
        elif beta is None:
            self.beta = 0.0
            self.annealing_beta = True
            self.anneal_cap = anneal_cap

    @property
    def _calculate_beta(self) -> float:
        if self.model.training and self.annealing_beta:
            self.beta = min(self.beta + self.anneal_amount, self.anneal_cap)
        return self.beta

    def _calculate_loss(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> float:
        softmax_output = F.log_softmax(output, dim=1)
        CE = -torch.mean(torch.sum(softmax_output * input, dim=1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return CE + self._calculate_beta * KLD

    def _train_step(self, user_batch: torch.Tensor) -> float:
        batch = torch.FloatTensor(self.sparse_train_input[user_batch[0]].toarray()).to(
            self.device
        )

        self.optimizer.zero_grad()

        output, mu, logvar = self.model.forward(batch)
        loss = self._calculate_loss(output, batch, mu, logvar)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def _validation_step(
        self, user_batch: torch.Tensor, calculate_loss: bool = False
    ) -> float:
        input = torch.FloatTensor(self.sparse_val_input[user_batch[0]].toarray()).to(
            self.device
        )
        label = torch.FloatTensor(self.sparse_val_label[user_batch[0]].toarray()).to(
            self.device
        )

        output, mu, logvar = self.model(input)

        output[input != 0] = -float("inf")
        flatten_output = output.flatten().cpu()
        flatten_label = label.flatten().cpu()
        user_index = (
            torch.arange(output.size(0), dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, output.size(1))
            .flatten()
        )

        ndcg = RetrievalNormalizedDCG(top_k=self.k)
        metric = ndcg(flatten_output, flatten_label, indexes=user_index).item()

        if calculate_loss:
            val_loss = self._calculate_loss(output, input, mu, logvar).item()
            return val_loss, metric

        return metric

    def save_checkpoint(self, path_to_save: str = None) -> None:
        if path_to_save is None:
            path_to_save = f"models/{self.dataset}/best_multivae.pth"

        torch.save(self.model.state_dict(), path_to_save)

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        train: csr_matrix,
        val: Dict[str, csr_matrix],
    ) -> None:
        self.best_ndcg_metric = 0.0
        self.best_ndcg_metric_for_beta = 0.0
        self.sparse_train_input = train
        self.sparse_val_input = val["input"]
        self.sparse_val_label = val["label"]

        for epoch in range(self.epochs_num):
            self.model.train()

            if self.set_lr_scheduler:
                self.lr_scheduler.step()

            average_train_loss = 0
            for batch_id, batch in enumerate(dataloader):
                train_loss = self._train_step(batch)
                average_train_loss += train_loss

            average_train_loss /= len(dataloader)
            Logger.current_logger().report_scalar(
                "Loss",
                "Train",
                value=average_train_loss,
                iteration=epoch,
            )
            print(f"Epoch = {epoch}: Loss={average_train_loss}")

            self.model.eval()
            with torch.no_grad():
                average_ndcg_metric = 0, 0
                for batch_id, batch in enumerate(dataloader):
                    current_ndcg_metric = self._validation_step(batch)

                    average_ndcg_metric += current_ndcg_metric

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

                Logger.current_logger().report_scalar(
                    "Validation metric",
                    "NDCG@10",
                    value=ndcg_metric,
                    iteration=epoch,
                )
                print(f"Epoch = {epoch}: NDCG@10 = {ndcg_metric}")

            if epoch == self.epochs_num - 1:
                self.save_checkpoint(f"models/{self.dataset}/final_multivae.pth")

    def _calculate_metrics(self, user_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        input = torch.FloatTensor(self.sparse_test_input[user_batch[0]].toarray()).to(
            self.device
        )
        label = torch.FloatTensor(self.sparse_test_label[user_batch[0]].toarray()).to(
            self.device
        )

        output, _, _ = self.model(input)

        output[input != 0] = -float("inf")
        flatten_output = output.flatten()
        flatten_label = label.flatten()
        user_index = (
            torch.arange(output.size(0), dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, output.size(1))
            .flatten()
        ).to(self.device)
        self.metrics.reset()

        return self.metrics(flatten_output, flatten_label, user_index)

    def test(
        self,
        dataloader: torch.utils.data.DataLoader,
        test: Dict[str, csr_matrix],
        ks: Sequence[int],
        path_to_load: str,
    ) -> None:
        self.sparse_test_input = test["input"]
        self.sparse_test_label = test["label"]

        if path_to_load is None:
            path_to_load = "models/{self.dataset}/best_multivae.pth"

        checkpoint = torch.load(path_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.ks = ks
        metrics = MetricCollection(
            [
                MetricCollection(
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

        self.metrics = metrics.to(self.device)
        # self.metrics = metrics

        self.model.eval()
        with torch.no_grad():
            result_metrics = {k: 0 for k in metrics.keys(keep_base=True)}

            for batch in dataloader:
                current_metrics = self._calculate_metrics(batch)
                result_metrics = {
                    k: result_metrics[k] + current_metrics[k] for k in result_metrics
                }

            result_metrics = {
                k: result_metrics[k] / len(dataloader) for k in result_metrics
            }

            for metric, value in result_metrics.items():
                Logger.current_logger().report_single_value(name=metric, value=value)
                print(f"{metric} = {value}")
