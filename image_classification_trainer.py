from base_trainer import BaseTrainer
from cfg_loader import ConfigLoader


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
        print(self.cfg)

    def build_model(self):
        return super().build_model()
    
    def load_model(self, path: str):
        return super().load_model(path)
    
    def save_model(self, path: str):
        return super().save_model(path)
    


if __name__ == '__main__':
   image_classifier = ImageClassificationTrainer("configs", "config")
   image_classifier.load_cfg()
   image_classifier.load_data()