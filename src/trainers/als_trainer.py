import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class AlSRecommender:
    def __init__(self, factors, regularization, alpha, iterations, use_gpu, random_state,):
        self.model = AlternatingLeastSquares(factors, regularization, alpha, iterations, use_gpu, random_state)
        self.model_trained = False
    
    def train(self, train: csr_matrix, save_model: bool = True):
        self.model.fit(train)
        self.model_trained = True

        if save_model:
            self.mode.save('als.npz')

    def _convert_to_df(self, recommendations_matrix: np.ndarray, recommendations_scores: np.ndarray):

        recommendations = pd.DataFrame({
            'user_id': np.arange(0, len(recommendations_matrix)),
            'item_id': list(recommendations_matrix),
            'score': list(recommendations_scores),
        }).explode(['item_id', 'score'], ignore_index=True)

        return recommendations
    
    def test(self, train, test, model_path='models/als/ml_1m.npz'):
        if not self.model_trained:
            self.model = self.model.load(model_path)

        recommendations_matrix, _ = self.model.recommend(
            train["user_id"].unique(),
            train,
            N=50,
            filter_already_liked_items=True
        )

        recommendations = pd.DataFrame({
            'user_id': np.arange(0, len(recommendations_matrix)),
            'recommended': list(recommendations_matrix.tolist()),
        })

        label = test.groupby('user_id')[['item_id']].agg(list).rename(columns={'item_id' : 'label'})

        labelled_recommendations = recommendations.merge(label, on='user_id', how='left')

        

