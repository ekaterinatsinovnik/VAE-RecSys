from typing import Tuple

import pandas as pd
from scipy.sparse import csr_matrix
from rectools.dataset import Dataset
from rectools import Columns

from .base_dataloader import BaseDataloader


class EASEDataLoader(BaseDataloader):
    def get_dataloaders(self) -> Tuple[csr_matrix, pd.DataFrame, int]:

        train_df, test_df = self._split()

        train_df.columns = [Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]

        train = Dataset.construct(train_df)

        return train, test_df, test_df.item_id.nunique()

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