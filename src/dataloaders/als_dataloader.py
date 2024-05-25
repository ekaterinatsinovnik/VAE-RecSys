from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .base_dataloader import BaseDataloader


class ALSDataLoader(BaseDataloader):
    def get_dataloaders(self, use_tfidf=False) -> Tuple[csr_matrix, pd.DataFrame, int]:

        train_df, test_df = self._split()

        sparse_train = self._convert_to_sparse(train_df)

        return sparse_train, test_df, sparse_train.shape[1]

    def _process_to_split(self, func_to_apply: callable) -> pd.DataFrame: 
        return (
            self.sorted_items_to_split.apply(func_to_apply, include_groups=False)
            .reset_index()
            .drop(columns=["level_1"])
        )

    def _split(self) -> callable:
        if isinstance(self.test_size, float):
            return self._split_by_ratio()
        else:
            return self._split_by_const()

    def _split_by_const(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = self._process_to_split(
                lambda row: row.iloc[:-self.test_size]
            )
        test = self._process_to_split(
                lambda row: row.iloc[-self.test_size:]
        )

        return train, test

    def _split_by_ratio(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def bound(row, ratio):
            return np.ceil(len(row) * ratio).astype("int")

        train = self._process_to_split(
            lambda row: row.iloc[: -bound(row, self.test_size)]
        )
        test = self._process_to_split(
            lambda row: row.iloc[-bound(row, self.test_size):]
        )

        return train, test

    @staticmethod
    def _encode_tfidf(interactions: pd.DataFrame) -> pd.DataFrame:

        rating_sum_per_user = interactions.groupby("user_id")["rating"].transform("sum")
        user_count_per_element = interactions.groupby("item_id")["user_id"].transform("size")
        
        tf = interactions["rating"].values / rating_sum_per_user.values
        idf = np.log(len(rating_sum_per_user) / user_count_per_element.values)

        tfidf_values = tf * idf

        return tfidf_values
    
    def _convert_to_sparse(self, interactions_df: pd.DataFrame, use_tfidf: bool = True) -> csr_matrix:
        if use_tfidf:
            sparse_matrix_values = self._encode_tfidf(interactions_df)
        else:
            sparse_matrix_values = np.ones(interactions_df.shape[0])

        user_index = interactions_df["user_id"].astype("category").cat.codes.values
        item_index = interactions_df["item_id"].astype("category").cat.codes.values
        assert len(user_index) == len(item_index)

        sparse_interactions = csr_matrix(
            (
                sparse_matrix_values,
                ([user_index, item_index]),
            ),
            shape=(self.unique_user_num, self.unique_item_num),
        )

        return sparse_interactions
