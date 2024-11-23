from base_trainer import BaseTrainer
from cfg_loader import ConfigLoader
import torch
import mlflow
from typing import Dict
from torch.utils.data import DataLoader


class ImageClassificationTrainer(BaseTrainer):
    def __init__(self, config_path, config_name) -> None:
        super().__init__()
        self.cfg = None
        self.cfg_path, self.cfg_name = config_path, config_name

    
    def load_cfg(self):
        self.cfg = ConfigLoader(self.cfg_path, self.cfg_name).load()
    
    def load_data(self):
        print(self.cfg)
        dataset_path = self.cfg.train.data_path
        print(dataset_path)
        return super().load_data()

    def train(self):
        epochs = self.cfg.train.epochs
        running_loss = 0.0
        for i in range(epochs):
            self.train_one_step()

    def train_one_step():
        pass

    def metrics(self) -> Dict:
        return {"Accuracy: ": 0.0,
                "Precision: ": 0.0,
                "Recall: ": 0.0,
                "F1-score: ": 0.0}

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
        
    


if __name__ == '__main__':
   image_classifier = ImageClassificationTrainer("configs", "config")
   image_classifier.load_cfg()
   image_classifier.load_data()