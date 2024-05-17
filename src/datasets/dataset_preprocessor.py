import os
import pickle
from typing import Dict, Tuple

import pandas as pd

DATASET_ROOT_PATH = "data"
PROCESSED_DATASET_PATH = "processed"


class DatasetPreprocessor:
    def __init__(
        self,
        dataset_shortname: str,
        min_rating: float = 3.5,
        min_user_count: int = 5,
        min_item_count: int = 0,
    ) -> None:
        self.dataset_shortname = dataset_shortname
        self.min_rating = min_rating
        self.min_user_count = min_user_count
        self.min_item_count = min_item_count

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

    def _filter(self) -> Tuple[pd.DataFrame, Dict[str, Dict[int, int]]]:
        interactions = self._load_raw_dataset()

        interactions = self._filter_by_min_rating(interactions)
        interactions = self._filter_triplets(interactions)

        interactions, user_mapping, item_mapping = self._encode_ids(interactions)

        mapping = {
            "user_mapping": user_mapping,
            "item_mapping": item_mapping,
        }

        return interactions, mapping

    def preprocess(self, processed_data_path) -> None:
        preprocessed_interactions, mapping = self._filter()
        os.mkdir(processed_data_path)

        preprocessed_interactions.to_parquet(
            os.path.join(processed_data_path, "interactions.parquet"), index=False
        )

        with open(os.path.join(processed_data_path, "mapping.pickle"), "wb") as f:
            pickle.dump(mapping, f)

    def load_dataset(self) -> pd.DataFrame:
        processed_data_path = os.path.join(self.data_path, PROCESSED_DATASET_PATH)

        if not os.path.exists(processed_data_path):
            self.preprocess(processed_data_path)

        interactions = pd.read_parquet(
            os.path.join(processed_data_path, "interactions.parquet")
        )
        # mapping = pd.read_pickle(os.path.join(processed_data_path, "mapping.pickle"))
        return interactions
