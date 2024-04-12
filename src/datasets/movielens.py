# import os

# import pandas as pd
# from base_dataset import BaseDataset


# class ML1m_Dataset(BaseDataset):
#     def _load_raw_dataset(self):
#         raw_data_path = os.path.join(self.data_path, "raw/ratings.dat")

#         raw_data = pd.read_csv(
#             raw_data_path,
#             header=0,
#             sep='::',
#             names=["user_id", "item_id", "rating", "timestamp"],
#         )

#         return raw_data



# class ML20m_Dataset(BaseDataset):

#     def _load_raw_dataset(self):
#         raw_data_path = os.path.join(self.data_path, "raw/ratings.csv")

#         raw_data = pd.read_csv(
#             raw_data_path,
#             header=0,
#             names=["user_id", "item_id", "rating", "timestamp"],
#         )

#         return raw_data

#     def _filter(self, raw_data):
#         raise NotImplementedError
