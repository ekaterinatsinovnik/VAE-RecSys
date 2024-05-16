from abc import ABC, abstractmethod


from src.datasets import DatasetPreprocessor


class BaseDataloader(ABC):
    def __init__(
        self,
        dataset_shortname: str,
        min_rating: float = 3.5,
        min_user_count: int = 5,
        min_item_count: int = 0,
    ) -> None:
        data = DatasetPreprocessor(
            dataset_shortname=dataset_shortname,
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
    def _convert_to_sparse(self):
        pass

    @abstractmethod
    def get_dataloaders(self):
        pass

    @abstractmethod
    def _split(self):
        pass
