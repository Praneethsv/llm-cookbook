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
    def save_model(self):
        """ Saves the trained model to a specific path. Model save path is taken from config.yaml """
        pass

    @abstractmethod
    def load_model(self):
        """ Loads the trained model given a specific path. Load path is taken from config.yaml """
        pass

    @abstractmethod
    def train_one_step():
        """Perform a single training step on a batch of data."""
        pass

    @abstractmethod
    def validate_one_step():
        """Perform a single validation step on a batch of data."""
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        """Evaluate the model on the given dataset."""
        pass
   
    @abstractmethod
    def setup_logger(self):
        """Set up the logger for training."""
        pass

