from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def load_cfg(self):
        pass

    @abstractmethod
    def load_data(self):
        """Load datasets for training, validation, and testing."""
        pass

    @abstractmethod
    def build_model(self):
        """Build or initialize the model."""
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_model(self, path: str):
        """ Saves the trained model to a specific path """
        pass

    @abstractmethod
    def load_model(self, path: str):
        """ Loads the trained model given a specific path """
        pass