from typing import Any

import torch.optim as optim
from omegaconf import OmegaConf


class Optimizer:
    def __init__(self, cfg) -> None:
        self.optim_cfg = cfg
        self.optimizers = optim.__all__
        self.optim_dict = {
            optim_name: getattr(optim, optim_name) for optim_name in self.optimizers
        }

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.get_optimizer()

    def get_optimizer(self, cfg=None):
        if cfg is None:
            cfg = self.optim_cfg

        for optimizer_name, config in cfg.items():

            if OmegaConf.is_dict(config) and config.get("enabled", False):

                return {self.optim_dict.get(optimizer_name, None): config}

        return {optim.Adam: {"lr": 3e-4}}
