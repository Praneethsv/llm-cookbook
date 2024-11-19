from omegaconf import DictConfig, OmegaConf
import hydra as H


class ConfigLoader:
    def __init__(self, cfg_dir: str, cfg_name) -> None:
        self.cfg_path = cfg_dir
        self.cfg_name = cfg_name
        self.config = None

    def load(self) -> DictConfig:
        with H.initialize(config_path=self.cfg_path):
            self.config = H.compose(config_name=self.cfg_name)

        return self.config