import os
import pickle
from typing import Dict, Tuple, Union

import pandas as pd

DATASET_ROOT_PATH = "data"
# PROCESSED_DATASET_PATH = "processed.pickle"
PROCESSED_DATASET_PATH = "processed"


class BaseDataset:
    def __init__(self, args: dict) -> None:
        self.dataset_shortname = args["dataset_shortname"]
        self.min_rating = args["min_rating"]
        self.min_user_count = args["min_user_count"]
        self.min_item_count = args["min_item_count"]

        self.data_path = self._get_data_common_path()

    def _load_raw_dataset(self) -> pd.DataFrame:
        raw_data_path = os.path.join(self.data_path, "raw/ratings.csv")

        raw_data = pd.read_csv(
            raw_data_path,
            header=0,
            names=["user_id", "item_id", "rating", "timestamp"],
        )

        return raw_data

    def _get_data_common_path(self) -> str:
        path = os.path.join(DATASET_ROOT_PATH, self.dataset_shortname)
        return path

    def _filter_by_min_rating(self, interactions: pd.DataFrame) -> pd.DataFrame:
        interactions = interactions[interactions.rating >= self.min_rating]
        return interactions

    def _get_count(self, interactions: pd.DataFrame, column_name: str) -> pd.Series:
        count_groups = interactions[[column_name]].groupby(column_name)
        count = count_groups.size()
        return count

    def _filter_triplets(self, interactions: pd.DataFrame) -> pd.DataFrame:
        if self.min_user_count > 0:
            user_count = self._get_count(interactions, "user_id")

            interactions = interactions[
                interactions["user_id"].isin(
                    user_count.index[user_count >= self.min_user_count]
                )
            ]

        if self.min_item_count > 0:
            item_count = self._get_count(interactions, "item_id")

            interactions = interactions[
                interactions["item_id"].isin(
                    item_count.index[item_count >= self.min_item_count]
                )
            ]

        return interactions

    def _encode_ids(self, interactions: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
        user_mapping = {u: i for i, u in enumerate(interactions["user_id"].unique())}
        item_mapping = {s: i for i, s in enumerate(interactions["item_id"].unique())}

        interactions["user_id"] = interactions["user_id"].map(user_mapping)
        interactions["item_id"] = interactions["item_id"].map(item_mapping)

        return interactions, user_mapping, item_mapping

    # def _split(
    #     self, interactions: pd.DataFrame, num_users: int
    # ) -> Tuple[dict, dict, dict]:

    #     sorted_items = interactions.groupby("user_id").apply(
    #         lambda d: list(d.sort_values(by="timestamp")["item_id"]),
    #         include_groups=False,
    #     )

    #     train, val, test = {}, {}, {}
    #     for user in range(num_users):
    #         items = sorted_items[user]
    #         train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]

    #     return train, val, test

    def _to_sparse_matrix(self, dataframe):
        pass

    def _filter(self) -> Dict[str, Union[pd.DataFrame, Dict[int, int]]]:
        interactions = self._load_raw_dataset()

        interactions = self._filter_by_min_rating(interactions)
        interactions = self._filter_triplets(interactions)

        interactions, user_mapping, item_mapping = self._encode_ids(interactions)

        # train, val, test = self._split(interactions, len(user_mapping))
        # dataset = {
        #     "train": train,
        #     "val": val,
        #     "test": test,
        #     "user_mapping": user_mapping,
        #     "item_mapping": item_mapping,
        # }

        mapping = {
            "user_mapping": user_mapping,
            "item_mapping": item_mapping,
        }

        return interactions, mapping

    def preprocess(self) -> None:
        processed_data_path = os.path.join(self.data_path, PROCESSED_DATASET_PATH)
        os.mkdir(processed_data_path)
        # if os.path.exists(processed_data_path):
        #     raise FileExistsError()  # should use it?

        preprocessed_interactions, mapping = self._filter()

        # with open(processed_data_path, "wb") as f:
        #     pickle.dump(preprocessed_data, f)

        preprocessed_interactions.to_parquet(
            os.path.join(processed_data_path, "interactions.parquet"), index=False
        )

        with open(os.path.join(processed_data_path, "mapping.pickle"), "wb") as f:
            pickle.dump(mapping, f)

    def load_dataset(self) -> Dict[str, Union[pd.DataFrame, Dict[int, int]]]:
        processed_data_path = os.path.join(self.data_path, PROCESSED_DATASET_PATH)

        if not os.path.exists(processed_data_path):
            self.preprocess()

        interactions = pd.read_parquet(
            os.path.join(processed_data_path, "interactions.parquet")
        )
        # mapping = pd.read_pickle(os.path.join(processed_data_path, "mapping.pickle"))
        return interactions
