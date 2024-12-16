import inspect
from typing import Dict

import torch.nn as nn
import torch.nn.modules.loss as loss
from omegaconf import OmegaConf

from cfg_loader import ConfigLoader
from utils.utils import get_enabled_configs


class Loss:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, *args, **kwds):
        return self.get_loss()

    def get_loss(self) -> Dict:
        enabled_loss = {}
        loss_cfg = get_enabled_configs(self.cfg, key="loss")
        loss_name = loss_cfg.get("name")
        enabled_loss_cls = getattr(loss, loss_name)
        valid_params = inspect.signature(enabled_loss_cls).parameters
        loss_cfg = {k: v for k, v in loss_cfg.items() if k in valid_params}
        enabled_loss[enabled_loss_cls] = loss_cfg
        return enabled_loss
