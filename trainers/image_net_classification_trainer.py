from typing import Dict

import mlflow
import torch

from base_trainer import BaseTrainer
from cfg_loader import ConfigLoader
from data_loaders.image_net_data_loader import ImageNetDataLoader


class ImageClassificationTrainer(BaseTrainer):
    def __init__(self, config_path, config_name) -> None:
        super().__init__()

        self.cfg_path, self.cfg_name = config_path, config_name
        self.cfg = self.load_cfg()
        image_net_data_loader = ImageNetDataLoader(self.cfg.train.data_loader)
        self.train_data, self.val_data, self.test_data = image_net_data_loader.split()

    def load_cfg(self):
        self.cfg = ConfigLoader(self.cfg_path, self.cfg_name).load()

    def train(self):
        epochs = self.cfg.train.epochs
        running_loss = 0.0
        for i in range(epochs):
            self.train_one_step()

    def train_one_step():
        pass

    def metrics(self) -> Dict:
        return {
            "Accuracy: ": 0.0,
            "Precision: ": 0.0,
            "Recall: ": 0.0,
            "F1-score: ": 0.0,
        }

    def build_model(self):
        return super().build_model()

    def load_model(self, path: str):
        return super().load_model(path)

    def save_model(self):
        model_saver_cfg = self.cfg.train.model_saver
        save_path = model_saver_cfg.path + model_saver_cfg.name
        torch.save(save_path)

    def setup_logger(self):
        with mlflow.start_run():
            mlflow.log_dict(self.metrics)


if __name__ == "__main__":
    image_classifier = ImageClassificationTrainer("configs", "config")
    image_classifier.load_cfg()
    image_classifier.load_data()
