from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader, TensorDataset

from src.datasets import BaseDataset


# class TrainLoader:
#     def __init__(self, )


class VAEDataLoader:
    # def __init__(self, train_batch_size, test_batch_size=None):
    #     self.train_batch_size = train_batch_size

    #     if test_batch_size is None:
    #         self.test_batch_size = train_batch_size
    #     else:
    #         self.test_batch_size = test_batch_size
    def __init__(self, batch_size):
        self.batch_size = batch_size

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

        # или возвращать в функции?
        # self.train_loader = self._get_dataloader(train, self.train_batch_size)
        # self.val_loader = self._get_dataloader(val, self.train_batch_size)
        # self.test_loader = self._get_dataloader(test, self.test_batch_size)

    def get_dataloaders(
        self,
        dataset_shortname: str,
        min_rating: float = 3.5,
        min_user_count: int = 5,
        min_item_count: int = 0,
    ):
        data = BaseDataset(
            args={
                "dataset_shortname": dataset_shortname,
                "min_rating": min_rating,
                "min_user_count": min_user_count,
                "min_item_count": min_item_count,
            }
        )

        interactions = data.load_dataset()

        self.unique_user_num = interactions["user_id"].nunique()
        self.unique_item_num = interactions["item_id"].nunique()

        train, val, test = self._split(interactions)

        # train_dataloader = self._get_train_dataloader(train)
        # val_dataloader = self._get_val_test_dataloader(val, validate=True)
        # test_dataloader = self._get_val_test_dataloader(test, validate=False)
        
        # return train_dataloader, val_dataloader, test_dataloader

        dataloader = DataLoader(
            TensorDataset(torch.arange(self.unique_user_num)),
            batch_size=self.batch_size,
            # batch_size=self.train_batch_size,
            shuffle=False,
        )

        sparse_train = self._get_train_dataloader(train)
        sparse_val = self._get_val_test_dataloader(val, validate=True)
        sparse_test = self._get_val_test_dataloader(test, validate=False)

        sparse_interactions = {'train' : sparse_train, 'val' : sparse_val, 'test' : sparse_test}

        # return {'item_num' : self.unique_item_num, 
        #     'dataloader' : dataloader, 
        # 
        return self.unique_item_num, dataloader, sparse_interactions

    # add val and test size (proportion)
    def _split(
        self, interactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        sorted_items = interactions.groupby("user_id").apply(
            lambda d: list(d.sort_values(by="timestamp")["item_id"]),
            include_groups=False,
        )

        train, val, test = (
            pd.DataFrame({"user_id": sorted_items.index}),
            pd.DataFrame({"user_id": sorted_items.index}),
            pd.DataFrame({"user_id": sorted_items.index}),
        )

        train["input_item_id"] = sorted_items.transform(lambda row: row[:-2])

        val["input_item_id"] = sorted_items.transform(lambda row: row[:-2])
        val["label_item_id"] = sorted_items.transform(lambda row: row[-2])

        test["input_item_id"] = sorted_items.transform(lambda row: row[:-1])
        test["label_item_id"] = sorted_items.transform(lambda row: row[-1])

        return train, val, test

    def _get_train_dataloader(self, interactions_grouped): 

        interactions_exploaded = interactions_grouped.explode("input_item_id")

        user_index = interactions_exploaded["user_id"].astype("category").cat.codes.values
        item_index = interactions_exploaded["input_item_id"].values
        assert len(user_index) == len(item_index)

        sparse_interactions = csr_matrix(
            (
                np.ones(interactions_exploaded.shape[0]),
                ([user_index, item_index]),
            ),
            shape=(self.unique_user_num, self.unique_item_num),
        )

        # dataloader = DataLoader(
        #     FloatTensor(sparse_interactions.toarray()),
        #     batch_size=self.train_batch_size,
        #     shuffle=True,
        # )
        
        return {'input' : sparse_interactions}

    def _get_val_test_dataloader(self, interactions_grouped, validate=True):
        user_label_index = (
            interactions_grouped["user_id"].astype("category").cat.codes.values
        )
        label_index = interactions_grouped["label_item_id"].values
        assert len(user_label_index) == len(label_index)

        interactions_exploaded = interactions_grouped.explode("input_item_id")

        user_input_index = (
            interactions_exploaded["user_id"].astype("category").cat.codes.values
        )
        input_index = interactions_exploaded["input_item_id"].values

        assert len(user_input_index) == len(input_index)

        sparse_input = csr_matrix(
            (
                np.ones(interactions_exploaded.shape[0]),
                ([user_input_index, input_index]),
            ),
            shape=(self.unique_user_num, self.unique_item_num),
        )

        sparse_label = csr_matrix(
            (
                np.ones(interactions_grouped.shape[0]),
                ([user_label_index, label_index]),
            ),
            shape=(self.unique_user_num, self.unique_item_num),
        )

        # if validate:
        #     batch_size = self.train_batch_size
        # else:
        #     batch_size = self.test_batch_size

        # dataloader = DataLoader(
        #     TensorDataset(
        #         FloatTensor(sparse_input.toarray()), FloatTensor(sparse_label.toarray())
        #     ),
        #     batch_size=batch_size,
        #     shuffle=False,
        # )

        return {'input' : sparse_input, 'label' : sparse_label}
