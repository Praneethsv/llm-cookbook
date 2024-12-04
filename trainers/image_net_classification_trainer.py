from typing import Dict

import mlflow
import torch
from model_zoo import ModelZoo

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
        self.model_zoo = ModelZoo(self.cfg.train.task)
        self.model = self.build_model(self.cfg)[0]

    def load_cfg(self):
        self.cfg = ConfigLoader(self.cfg_path, self.cfg_name).load()
        return self.cfg

    def train(self):
        epochs = self.cfg.train.epochs
        running_loss = 0.0
        for i in range(epochs):
            self.train_one_step()

    def train_one_step(self):
        self.model.train()  # to set the model mode to train

    def metrics(self) -> Dict:
        return {
            "Accuracy: ": 0.0,
            "Precision: ": 0.0,
            "Recall: ": 0.0,
            "F1-score: ": 0.0,
        }

    def build_model(self, task_cfg):
        models_dict = self.model_zoo.get_models(task_cfg)
        models = []
        for model_instance, model_cfg in models_dict.items():
            models.append(model_instance(**model_cfg))
        return models

    def load_model(self, path: str):
        return super().load_model(path)

    def save_model(self):
        model_saver_cfg = self.cfg.train.model_saver
        save_path = model_saver_cfg.path + model_saver_cfg.name
        torch.save(save_path)

    def validate_one_step():
        pass

    def evaluate(self, data_loader):
        return super().evaluate(data_loader)

    def setup_logger(self):
        with mlflow.start_run():
            mlflow.log_dict(self.metrics)


if __name__ == "__main__":
    image_classifier = ImageClassificationTrainer("configs", "config")
    image_classifier.load_cfg()
    image_classifier.train()
