import importlib
import inspect
import pkgutil
from collections import defaultdict

from omegaconf import OmegaConf

import models


class ModelZoo:

    def __init__(self, cfg) -> None:
        self.batch_size = cfg.train.data_loader.batch_size
        self.task_cfg = cfg.train.task
        self.model_classes = self.register_models("models")
        self.tasks = None
        self.models = None

    def get_models(self, cfg, path="", models=None):
        """Depending on the task, retrieves the class for building a model"""
        if models is None:
            self.models = []
            self.model_cfgs = []
        if OmegaConf.is_dict(cfg):
            if cfg.get("enabled", False):
                model = cfg.get("model", "")
                if model:
                    self.models.append(model.name)
                    self.model_cfgs.append(model)

            for key, value in cfg.items():
                if OmegaConf.is_dict(value):
                    self.get_models(
                        value, f"{path}.{key}" if path else key, self.models
                    )
        model_classes = {}
        for model_name, model_cfg in zip(self.models, self.model_cfgs):
            model = self.get_class_by_name(model_name)
            if model not in model_classes.keys():
                model_classes[model] = model_cfg
        return model_classes

    def get_class_by_name(self, name):
        for key, classes in self.model_classes.items():
            for cls in classes:
                if name.lower() in cls.__name__.lower():  # case-insensitive matching
                    return cls
        return None

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
