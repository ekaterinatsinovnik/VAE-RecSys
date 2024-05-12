import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from .base_dataloader import BaseDataloader


class ALSLoader(BaseDataloader):
    def make_sparse(self, grouped_ratings):
        ratings = grouped_ratings.explode()

        user_encoder = LabelEncoder()
        user_index = user_encoder.fit_transform(ratings.index.to_numpy())

        item_encoder = LabelEncoder()
        item_index = item_encoder.fit_transform(ratings.values)

        user_num = len(grouped_ratings)
        item_num = len(np.unique(item_index))

        sparse_matrix = csr_matrix(
            (np.ones(len(user_index)), (user_index, item_index)),
            shape=(user_num, item_num),
        )

        return sparse_matrix, user_encoder, item_encoder

    def load_dataset(self):
        train_csr, user_encoder, item_encoder = self.make_sparse(self.train)
        ground_truth = self.val.explode()
        return train_csr, user_encoder, item_encoder, ground_truth
