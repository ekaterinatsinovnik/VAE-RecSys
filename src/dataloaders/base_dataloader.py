from abc import ABC, abstractmethod


class BaseDataloader(ABC):
    def __init__(self, dataset): 
        
