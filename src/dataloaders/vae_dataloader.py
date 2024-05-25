from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset

from .base_dataloader import BaseDataloader


class VAEDataLoader(BaseDataloader):
    def __init__(
        self,
        dataset: str,
        min_rating: float,
        min_user_count: int,
        min_item_count: int,
        batch_size: int = 0,
        val_size: Union[float, int] = 0.1,
        test_size: Union[float, int] = 0.1,
    ) -> None:
        super().__init__(dataset, min_rating, min_user_count, min_item_count)
        self.batch_size = batch_size

    def get_dataloaders(self) -> Tuple[DataLoader, dict, int]:
        dataloader = DataLoader(
            TensorDataset(torch.arange(self.unique_user_num)),
            batch_size=self.batch_size,
            shuffle=False
        )

        train_val_input, val_label, test_input, test_label = self._split()

        sparse_train = self._convert_to_sparse(train_val_input)

        sparse_val = {"input": sparse_train, "label": self._convert_to_sparse(val_label)}

        sparse_test = {
            "input": self._convert_to_sparse(test_input),
            "label": self._convert_to_sparse(test_label),
        }

        sparse_interactions = {
            "train": sparse_train,
            "val": sparse_val,
            "test": sparse_test,
        }

        return dataloader, sparse_interactions, sparse_train.shape[1]

    def _convert_to_sparse(self, interactions_df: pd.DataFrame) -> csr_matrix:
        user_index = interactions_df["user_id"].astype('category').cat.codes.values
        item_index = interactions_df["item_id"].astype('category').cat.codes.values
        assert len(user_index) == len(item_index)

        sparse_interactions = csr_matrix(
            (
                np.ones(interactions_df.shape[0]),
                ([user_index, item_index]),
            ),
            shape=(self.unique_user_num, self.unique_item_num),
        )

        return sparse_interactions

    def _process_to_split(self, func_to_apply: callable) -> pd.DataFrame:
        return (
            self.sorted_items_to_split.apply(func_to_apply, include_groups=False)
            .reset_index()
            .drop(columns=["level_1"])
        )

    def _split(self) -> callable:
        if isinstance(self.val_size, float):
            return self._split_by_ratio()

        else:
            return self._split_by_const()

    def _split_by_ratio(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def bound(row, ratio):
            return np.ceil(len(row) * ratio).astype("int")

        self.val_size += self.test_size

        train_val_input = self._process_to_split(
            lambda row: row.iloc[: -bound(row, self.val_size)]
        )

        val_label = self._process_to_split(
            lambda row: row.iloc[-bound(row, self.val_size) : -bound(row, self.test_size)]
        )

        test_input = self._process_to_split(
            lambda row: row.iloc[: -bound(row, self.test_size)]
        )

        test_label = self._process_to_split(
            lambda row: row.iloc[-bound(row, self.test_size) :]
        )

        return train_val_input, val_label, test_input, test_label

    def _split_by_const(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.val_size += self.test_size

        train_val_input = self._process_to_split(lambda row: row.iloc[: -self.val_size])

        val_label = self._process_to_split(
            lambda row: row.iloc[-self.val_size : -self.test_size]
        )

        test_input = self._process_to_split(lambda row: row.iloc[: -self.test_size])

        test_label = self._process_to_split(lambda row: row.iloc[-self.test_size :])

        return train_val_input, val_label, test_input, test_label
