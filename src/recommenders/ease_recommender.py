import pandas as pd
from rectools import Columns
from rectools.models import EASEModel

from .metrics import compute_metrics
from .base_recommender import BaseRecommender


class EASERecommender(BaseRecommender):
    def __init__(
        self,
        dataset,
        regularization=500.0,
        num_threads=10,
    ):
        self.model = EASEModel(regularization, num_threads)
        self.model_trained = False
        self.dataset = dataset

    def train(self, train):
        self.model.fit(train)
        self.model_trained = True

    def test(self, train_data, test_data, ks):
        if not self.model_trained:
            self.model = self.model.train(train_data)

        recommendations = self.model.recommend(
            train_data[Columns.User].unique(),
            train_data,
            N=50,
            filter_already_liked_items=True,
        )

        recommendations = (
            recommendations.groupby("user_id")[["item_id"]]
            .agg(list)
            .rename(columns={"item_id": "recommended"})
        )

        label = (
            test_data.groupby("user_id")[["item_id"]]
            .agg(list)
            .rename(columns={"item_id": "label"})
        )

        labelled_recommendations = recommendations.merge(label, on="user_id", how="left")

        metrics = compute_metrics(labelled_recommendations, ks)

        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(f"{self.dataset}/ease_metrics.csv")
