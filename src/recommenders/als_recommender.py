import os
from typing import Union

import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from .metrics import compute_metrics
from .base_recommender import BaseRecommender


class ALSRecommender(BaseRecommender):
    def __init__(
        self,
        dataset: str,
        factors:int = 100,
        regularization: float = 0.01,
        alpha: float = 1.0,
        iterations: int = 15,
        random_state: int = 0,
        model_path: Union[str, None] = None
    ) -> None:

        self.model = AlternatingLeastSquares(
            factors, regularization, alpha, iterations, random_state
        )
        self.model_trained = False
        self.dataset = dataset

        if model_path is None:
            model_path = f"models/{self.dataset}"

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model_path = os.path.join(model_path, "als.npz")

    def train(self, train: csr_matrix):
        self.model.fit(train)
        self.model_trained = True

        if self.model_path is not None:
            self.mode.save(self.model_path)

    def test(self, train, test, ks):
        if not self.model_trained:
            self.model = self.model.load(self.model_path)

        recommendations_matrix, _ = self.model.recommend(
            train["user_id"].unique(), train, N=max(ks), filter_already_liked_items=True
        )

        recommendations = pd.DataFrame(
            {
                "user_id": train["user_id"].unique(),
                "recommended": list(recommendations_matrix.tolist()),
            }
        )

        label = (
            test.groupby("user_id")[["item_id"]]
            .agg(list)
            .rename(columns={"item_id": "label"})
        )

        labelled_recommendations = recommendations.merge(label, on="user_id", how="left")

        metrics = compute_metrics(labelled_recommendations, ks)

        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(os.join(self.model_path, "metrics.csv"))
