from abc import ABC, abstractmethod


class BaseDataLoader(ABC):

    @abstractmethod
    def load_data(self):
        """ Loads the raw dataset from a source such as a file or a database"""
        pass

    @abstractmethod
    def split(self):
        """ Splits the dataset into train, validation, and test samples"""
        pass

    @abstractmethod
    def preprocess(self):
        pass
    
    @abstractmethod
    def get_batch(self):
        pass

    @abstractmethod
    def get_labels(self):
        pass


