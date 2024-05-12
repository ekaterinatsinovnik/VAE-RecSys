import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from clearml import Logger
from torchmetrics.retrieval.ndcg import RetrievalNormalizedDCG

from src.models import MultiVAE


class MultiVAETrainer:
    def __init__(
        self,
        item_num,
        latent_dim=200,
        num_hidden=1,
        hidden_dim=600,
        dropout=0.1,
        epochs_num=10,
        learning_rate=2e-3,
        weight_decay=3e-6,
        k=10,
        beta=0.2,
        total_anneal_steps=2000,
        anneal_cap=1.0,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # или перенести создание модели в методы: типа для трейна так, для теста проверка на наличие модели после трейна, потом

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.model = MultiVAE(item_num, latent_dim, num_hidden, hidden_dim, dropout).to(
            self.device
        )

        self.learning_rate = learning_rate
        self.epochs_num = epochs_num

        # self.anneal_coef = config['anneal_coef'
        # self.l2_reg = l2_reg
        # self.factor = factor
        # self.patience = patience

        # функция гет оптимизер?
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.k = k

        if isinstance(beta, (float, int)):
            self.beta = beta  # мб не так
            self.anneal_cap = anneal_cap
            self.finding_best_beta = False
        elif beta is None:
            self.beta = 0.0
            self.finding_best_beta = True
            self.current_best_metric = 0.0
            self.anneal_cap = 1.0
        else:
            # или просто дефолтом поставить а здесь проверку на тип? или проверку на тип не здесь, а в мейне?
            self.beta = 0.2  # подобрать
            self.anneal_cap = anneal_cap  # заменить или убрать?

        self.anneal_amount = 1.0 / total_anneal_steps

        # self.beta = 0.0
        # self.finding_best_beta = (
        #     find_best_beta  # заменить на если бета None, то вычисляем
        # )
        # self.anneal_amount = 1.0 / total_anneal_steps

        # if self.finding_best_beta:
        #     self.current_best_metric = 0.0
        #     self.anneal_cap = 1.0
        # else:
        #     self.anneal_cap = anneal_cap

    @property
    def _calculate_beta(self):
        if self.model.training:
            self.beta = min(self.beta + self.anneal_amount, self.anneal_cap)
        return self.beta

    def _calculate_loss(self, output, input, mu, logvar):
        softmax_output = F.log_softmax(output, dim=1)
        CE = -torch.mean(torch.sum(softmax_output * input, dim=1))  # расшифровать се
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return CE + self._calculate_beta * KLD

    def _train_step(self, batch):
        # add lr scheduler
        batch = batch.to(self.device)
        self.optimizer.zero_grad()

        output, mu, logvar = self.model.forward(batch)
        loss = self._calculate_loss(output, batch, mu, logvar)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def _validation_step(self, batch):
        input, label = batch  # попробовать без девайс еще
        input, label = input.to(self.device), label.to(self.device)
        output, mu, logvar = self.model(input)  # мб сделать без возращения мю и вар

        val_loss = self._calculate_loss(output, input, mu, logvar).item()
        # output = output.to("cpu")

        # output[input != 0] = -float("inf") # IMPORTANT: remove items that were in the input
        flatten_output = output.flatten().cpu()
        flatten_label = label.flatten().cpu()
        user_index = (
            torch.arange(output.size(0), dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, output.size(1))
            .flatten()
        )

        metric = RetrievalNormalizedDCG(top_k=self.k)(
            flatten_output, flatten_label, indexes=user_index
        ).item()

        return val_loss, metric

    # https://medium.com/@a0922/introduction-to-clearml-executing-in-google-colab-5485768cf6e9
    def save_checkpoint(self):
        # ckpt_path = (
        #     cache_path
        #     / f"model_{model.info()}_optim_{optimizer_name}_acc_{accuracy:.4f}_epoch_{epoch_id}.ckpt"
        # )
        # torch.save(self.model.state_dict(), os.path.join(gettempdir(), "multivae.pt"))
        print(f"***\n\nDir to dave model {os.getcwd()}")
        # torch.save(self.model.state_dict(), "multivae.pt")
        # torch.save(self.model.state_dict(), os.path.join(os.getcwd(), "multivae.pt"))
        torch.save(self.model.state_dict(), os.path.join("models/ml-1m", "multivae.pt"))

    def train(self, train_loader, val_loader):
        self.best_ndcg_metric = float("-inf")

        for epoch in range(self.epochs_num):
            self.model.train()

            average_train_loss = 0
            for batch_id, batch in enumerate(train_loader):
                # batch = [x.to(self.device) for x in batch]
                train_loss = self._train_step(batch)
                average_train_loss += train_loss

            Logger.current_logger().report_scalar(
                "Loss",
                "Train",
                value=average_train_loss / len(train_loader),
                iteration=epoch,
            )
            # print(
            #     "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            #         epoch,
            #         batch_id,
            #         len(train_loader),
            #         100.0 * batch_id / len(train_loader),
            #         train_loss,
            #     )
            # )
            # train_debug_message = f"""Epoch[{epoch}] current loss:
            #                     {train_loss:.5f}"""
            # self.logger.debug(train_debug_message)

            self.model.eval()
            with torch.no_grad():
                # tqdm_dataloader = tqdm(self.val_loader)
                average_val_loss, average_ndcg_metric = 0, 0
                for batch_id, batch in enumerate(val_loader):
                    # batch = [x.to(self.device) for x in batch]

                    current_val_loss, current_ndcg_metric = self._validation_step(batch)

                    average_val_loss += current_val_loss
                    average_ndcg_metric += current_ndcg_metric

                    if self.finding_best_beta:
                        self.best_beta = self.beta
                        Logger.current_logger().report_scalar(
                            "Annealing beta",
                            "beta",
                            value=self.best_beta,
                            iteration=(epoch * len(val_loader) + batch_id),
                        )

                val_loss = average_val_loss / len(val_loader)
                ndcg_metric = average_ndcg_metric / len(val_loader)

                if self.best_ndcg_metric < ndcg_metric:
                    self.best_ndcg_metric = ndcg_metric
                    self.save_checkpoint()

                Logger.current_logger().report_scalar(
                    "Loss",
                    "Validation",
                    value=val_loss,
                    iteration=epoch,
                )

                Logger.current_logger().report_scalar(
                    "Validation metric",
                    "NDCG@10",
                    value=ndcg_metric,
                    iteration=epoch,
                )

                # print(
                #     "Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\Metric: {:.6f}".format(
                #         epoch,
                #         batch_id,
                #         len(val_loader),
                #         100.0 * batch_id / len(val_loader),
                #         val_loss,
                #         ndcg_metric,
                #     )
                # )
                # logging code

        # valid_loss = 0
        # with torch.no_grad():
        #     for batch in valid_data_loader:
        #         model_result = self._batch_pass(batch, self.model)
        #         valid_loss += self._loss(**model_result)
        #     valid_loss /= len(valid_data_loader)
        #     valid_debug_message = f"""Epoch[{epoch}] validation
        #                             average loss: {valid_loss:.5f}"""
        #     self.logger.debug(valid_debug_message)
        # return valid_loss.item()

        ### в исходном коде сравнивают ндсж и сохраняют, если скор на валидации лучше.
        # в реплее сохраняют по лоссу на валидации (вроде менее логично, это же реконструкция)
        # в корецском сохраняют все, при подсчете метрик на валидации сохраняют бету
