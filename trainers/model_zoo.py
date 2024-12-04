import importlib
import inspect
import pkgutil
from collections import defaultdict
from dataclasses import dataclass

from omegaconf import OmegaConf

import models
from cfg_loader import ConfigLoader


class ModelZoo:

    def __init__(self, task_cfg) -> None:
        self.task_cfg = task_cfg
        self.model_classes = self.register_models("models")
        self.tasks = None
        self.models = None

    def get_models(self, cfg, path="", models=None):
        """Depending on the task, retrieves the class for building a model"""
        if models is None:
            self.models = []
        if OmegaConf.is_dict(cfg):
            if cfg.get("enabled", False):
                model = cfg.get("model", "")
                if model:
                    self.models.append(model.name)

            for key, value in cfg.items():
                if OmegaConf.is_dict(value):
                    self.get_models(
                        value, f"{path}.{key}" if path else key, self.models
                    )
        model_classes = []
        for model_name, module_classes in self.model_classes.items():
            for module_class in module_classes:
                if module_class.__name__ in self.models:
                    model_classes.append(module_class)
        return model_classes

    def register_models(self, pkg_name):
        pkg_classes = defaultdict(list)
        package = importlib.import_module(pkg_name)
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            try:
                module = importlib.import_module(module_name)
                classes = [
                    obj
                    for name, obj in inspect.getmembers(module, inspect.isclass)
                    if obj.__module__ == module_name
                ]
                if classes:
                    pkg_classes[module_name].extend(classes)
            except Exception as e:
                print(f"Could not import {module_name}: {e}")

        return pkg_classes
