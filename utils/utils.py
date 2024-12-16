import importlib
import inspect
import pkgutil
from collections import defaultdict

from omegaconf import OmegaConf


def get_enabled_configs(config, key=None):

    def traverse_and_find_enabled(config, target_key):
        if OmegaConf.is_dict(config):
            if target_key in config:
                return config[target_key] if config.get("enabled", False) else {}
            for sub_key, sub_config in config.items():
                if OmegaConf.is_dict(sub_config):
                    result = traverse_and_find_enabled(sub_config, target_key)
                    if result:
                        return result
        return {}

    if key:
        return traverse_and_find_enabled(config, key)

    enabled_configs = {}

    for section, section_config in config.items():
        if OmegaConf.is_dict(section_config) and section_config.get("enabled", False):
            enabled_configs[section] = section_config

    return enabled_configs


def get_all_modules(pkg_name):
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
