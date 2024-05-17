import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from .metrics import compute_metrics


class ALSRecommender:
    def __init__(
        self,
        dataset_shortname,
        factors = 100,
        regularization= 0.01,
        alpha=1.0,
        iterations= 15,
        random_state=7,
 
    ):
        self.model = AlternatingLeastSquares(
            factors, regularization, alpha, iterations, random_state
        )
        self.model_trained = False
        self.dataset_shortname = dataset_shortname

    def train(self, train: csr_matrix, save_model: bool = True):
        self.model.fit(train)
        self.model_trained = True

        if save_model:
            self.mode.save("als.npz")

    def test(self, train, test, ks, model_path="models/als/ml_1m.npz"):
        if not self.model_trained:
            self.model = self.model.load(model_path)

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
        metrics_df.to_csv(f"als_{self.dataset_shortname}_metrics.csv")
