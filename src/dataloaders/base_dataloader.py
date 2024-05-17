from abc import ABC, abstractmethod
from typing import Union

from src.datasets import DatasetPreprocessor


class BaseDataloader(ABC):
    def __init__(
        self,
        dataset: str,
        min_rating: float = 3.5,
        min_user_count: int = 5,
        min_item_count: int = 0,
        val_size: Union[float, int] = 1,
        test_size: Union[float, int] = 1,
    ) -> None:

        self.val_size = val_size + test_size
        self.test_size = test_size

        data = DatasetPreprocessor(
            dataset_shortname=dataset,
            min_rating=min_rating,
            min_user_count=min_user_count,
            min_item_count=min_item_count,
        )

        interactions = data.load_dataset()
        
        self.sorted_items_to_split = interactions.sort_values(
            by=["user_id", "timestamp"], ascending=True
        ).groupby("user_id")

        self.unique_user_num = interactions["user_id"].nunique()
        self.unique_item_num = interactions["item_id"].nunique()

    @abstractmethod
    def get_dataloaders(self):
        pass

    @abstractmethod
    def _split(self):
        pass
