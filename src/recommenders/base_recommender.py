from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod    
    def test(self):
        pass
