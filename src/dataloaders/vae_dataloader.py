from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset

from .base_dataloader import BaseDataloader


class VAEDataLoader(BaseDataloader):
    # def __init__(self, train_batch_size, test_batch_size=None):
    #     self.train_batch_size = train_batch_size

    #     if test_batch_size is None:
    #         self.test_batch_size = train_batch_size
    #     else:
    #         self.test_batch_size = test_batch_size
    def __init__(
        self,
        dataset_shortname: str,
        min_rating: float,
        min_user_count: int,
        min_item_count: int,
        batch_size: int = 0,
        val_size: Union[float, int] = 1,
        test_size: Union[float, int] = 1,
    ) -> None:
        super().__init__(dataset_shortname, min_rating, min_user_count, min_item_count)
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size

        # # For autoencoders, we should remove users from val/test sets
        # # that rated items NOT in the training set

        # # extract a list of unique items from the training set
        # unique_items = set()
        # for items in self.train.values():
        #     unique_items.update(items)

        # self.unique_item_ids = interactions['item_id'].unique()
        # self.unique_user_ids = interactions['user_id'].unique()

        # self.unique_user_num = len(self.unique_user_ids)
        # self.unique_item_num = len(self.unique_item_ids)

    def get_dataloaders(self) -> Tuple[DataLoader, dict]:
        dataloader = DataLoader(
            TensorDataset(torch.arange(self.unique_user_num)),
            batch_size=self.batch_size,
            shuffle=False,
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

        return dataloader, sparse_interactions

    def _convert_to_sparse(self, interactions_df) -> csr_matrix:
        user_index = interactions_df["user_id"].astype("category").cat.codes.values
        item_index = interactions_df["item_id"].values
        assert len(user_index) == len(item_index)

        sparse_interactions = csr_matrix(
            (
                np.ones(interactions_df.shape[0]),
                ([user_index, item_index]),
            ),
            shape=(self.unique_user_num, self.unique_item_num),
        )

        return sparse_interactions

    def _process_to_split(self, func_to_apply):
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

    # add val and test size (proportion)
    # def _split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    #     sorted_items = self.interactions.groupby("user_id").apply(
    #         lambda d: list(d.sort_values(by="timestamp")["item_id"]),
    #         include_groups=False,
    #     )

    #     train, val, test = (
    #         pd.DataFrame({"user_id": sorted_items.index}),
    #         pd.DataFrame({"user_id": sorted_items.index}),
    #         pd.DataFrame({"user_id": sorted_items.index}),
    #     )

    #     train["input_item_id"] = sorted_items.transform(lambda row: row[:-2])

    #     val["input_item_id"] = sorted_items.transform(lambda row: row[:-2])
    #     val["label_item_id"] = sorted_items.transform(lambda row: row[-2])

    #     test["input_item_id"] = sorted_items.transform(lambda row: row[:-1])
    #     test["label_item_id"] = sorted_items.transform(lambda row: row[-1])

    #     return train, val, test

    # def _get_train_dataloader(self, interactions_grouped):

    #     interactions_exploaded = interactions_grouped.explode("input_item_id")

    #     user_index = interactions_exploaded["user_id"].astype("category").cat.codes.values
    #     item_index = interactions_exploaded["input_item_id"].values
    #     assert len(user_index) == len(item_index)

    #     sparse_interactions = csr_matrix(
    #         (
    #             np.ones(interactions_exploaded.shape[0]),
    #             ([user_index, item_index]),
    #         ),
    #         shape=(self.unique_user_num, self.unique_item_num),
    #     )

    #     return {'input' : sparse_interactions}

    # def _get_val_test_dataloader(self, interactions_grouped):
    #     user_label_index = (
    #         interactions_grouped["user_id"].astype("category").cat.codes.values
    #     )
    #     label_index = interactions_grouped["label_item_id"].values
    #     assert len(user_label_index) == len(label_index)

    #     interactions_exploaded = interactions_grouped.explode("input_item_id")

    #     user_input_index = (
    #         interactions_exploaded["user_id"].astype("category").cat.codes.values
    #     )
    #     input_index = interactions_exploaded["input_item_id"].values

    #     assert len(user_input_index) == len(input_index)

    #     sparse_input = csr_matrix(
    #         (
    #             np.ones(interactions_exploaded.shape[0]),
    #             ([user_input_index, input_index]),
    #         ),
    #         shape=(self.unique_user_num, self.unique_item_num),
    #     )

    #     sparse_label = csr_matrix(
    #         (
    #             np.ones(interactions_grouped.shape[0]),
    #             ([user_label_index, label_index]),
    #         ),
    #         shape=(self.unique_user_num, self.unique_item_num),
    #     )

    #     return {'input' : sparse_input, 'label' : sparse_label}
