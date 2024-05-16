from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .base_dataloader import BaseDataloader


class ALSDataLoader(BaseDataloader):
    def get_dataloaders(self) -> Tuple[csr_matrix, pd.DataFrame]:
        self.unique_user_num = self.interactions["user_id"].nunique()
        self.unique_item_num = self.interactions["item_id"].nunique()

        train_df, test_df = self._split()

        sparse_train = self._convert_to_sparse(train_df)

        return sparse_train, test_df

    def _split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = (
            self.sorted_items_to_split.apply(
                lambda row: row.iloc[:-1], include_groups=False
            )
            .reset_index()
            .drop(columns=["level_1"])
        )

        test = (
            self.sorted_items_to_split.apply(
                lambda row: row.iloc[-1:], include_groups=False
            )
            .reset_index()
            .drop(columns=["level_1"])
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
    
    def _convert_to_sparse(self, interactions_df, use_tfidf=True) -> csr_matrix:
        if use_tfidf:
            sparse_matrix_values = self._encode_tfidf(interactions_df)
        else:
            sparse_matrix_values = np.ones(interactions_df.shape[0])

        user_index = interactions_df["user_id"].astype("category").cat.codes.values
        item_index = interactions_df["item_id"].values
        assert len(user_index) == len(item_index)

        sparse_interactions = csr_matrix(
            (
                sparse_matrix_values,
                ([user_index, item_index]),
            ),
            shape=(self.unique_user_num, self.unique_item_num),
        )

        return sparse_interactions
