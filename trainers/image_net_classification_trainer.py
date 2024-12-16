from typing import Dict

import mlflow
import torch
import torch.nn as nn
from model_zoo import ModelZoo

from base_trainer import BaseTrainer
from cfg_loader import ConfigLoader
from data_loaders.image_net_data_loader import ImageNetDataLoader
from loss import Loss
from optimizer import Optimizer


class ImageNetClassificationTrainer(BaseTrainer):
    def __init__(self, config_path, config_name) -> None:
        super().__init__()

        self.cfg_path, self.cfg_name = config_path, config_name
        self.cfg = self.load_cfg()
        self.image_net_data_loader = ImageNetDataLoader(self.cfg.train.data_loader)
        self.train_data, self.val_data, self.test_data = (
            self.image_net_data_loader.split()
        )
        self.model_zoo = ModelZoo(self.cfg.train.task)
        self.device = self.cfg.train.device
        (self.model,) = self.build_model(self.cfg, self.device)
        self.optimizer_dict = Optimizer(self.cfg.train.optimizer)()
        self.optimizer_cls, self.optimizer_cfg = next(iter(self.optimizer_dict.items()))
        self.optimizer = self.optimizer_cls(
            self.model.parameters(), **self.optimizer_cfg
        )
        self.loss_fn = Loss(self.cfg.train.task)()
        self.criterion, self.loss_cfg = next(iter(self.loss_fn.items()))
        self.batch_size = self.cfg.train.data_loader.batch_size

    def load_cfg(self):
        self.cfg = ConfigLoader(self.cfg_path, self.cfg_name).load()
        return self.cfg

    def train(self):
        epochs = self.cfg.train.epochs
        running_loss = 0.0
        loss_criterion = self.criterion(**self.loss_cfg)
        print(loss_criterion)
        num_iterations = len(self.train_data) // self.batch_size
        print(num_iterations)
        for i in range(epochs):
            curr_batch_idx = 0
            for it in range(num_iterations):
                batch_idx = curr_batch_idx + self.batch_size - 1
                batch_images, batch_labels = self.image_net_data_loader.get_batch(
                    self.train_data, batch_idx=batch_idx
                )
                self.train_one_step(batch_images, batch_labels)

    def train_one_step(self, batch_images: torch.tensor, batch_labels: torch.tensor):
        self.model.train()  # to set the model mode to train
        out = self.model(batch_images)
        loss = out - batch_labels
        loss.backward()
        self.optimizer.step()

    def metrics(self) -> Dict:
        return {
            "Accuracy: ": 0.0,
            "Precision: ": 0.0,
            "Recall: ": 0.0,
            "F1-score: ": 0.0,
        }

    def build_model(self, task_cfg, device="cpu"):
        models_dict = self.model_zoo.get_models(task_cfg)
        models = []
        for model_instance, model_cfg in models_dict.items():
            models.append(model_instance(**model_cfg).to(device))
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
    image_classifier = ImageNetClassificationTrainer("configs", "config")
    image_classifier.train()
