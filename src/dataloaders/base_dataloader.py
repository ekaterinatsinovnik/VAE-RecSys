from abc import ABC, abstractmethod


class BaseDataloader(ABC):
    def __init__(self, dataset): # add random seed
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.user_mapping = dataset['user_mapping']
        self.item_mapping = dataset['item_mapping']
        self.user_num = len(self.user_mapping)
        self.item_num = len(self.item_mapping)
