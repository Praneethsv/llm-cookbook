from cfg_loader import ConfigLoader



class Trainer:
    def __init__(self, config_path, config_name) -> None:
        self.cfg_path = config_path
        self.cfg_name = config_name

    def load_cfg(self):
        self.cfg = ConfigLoader(self.cfg_path, self.cfg_name).load()
        print(self.cfg)
        print(self.cfg.train.optimizer)
        print(self.cfg.train.device)


    def train(self):
        pass



if __name__ == '__main__':
    trainer = Trainer("configs", "config")
    trainer.load_cfg()