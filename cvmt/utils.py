"""Shared utilities that are used by data and/or ml modules."""

from typing import Any, Dict

from easydict import EasyDict
import yaml
import shutil
import os


def nested_dict_to_easydict(nested_dict: Dict) -> EasyDict:
    if isinstance(nested_dict, dict):
        for key, value in nested_dict.items():
            nested_dict[key] = nested_dict_to_easydict(value)
        return EasyDict(nested_dict)
    elif isinstance(nested_dict, list):
        return [nested_dict_to_easydict(item) for item in nested_dict]
    else:
        return nested_dict


def load_yaml_params(path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def remove_lightning_logs_dir():
    path = "lightning_logs"
    if os.path.exists(path):
        shutil.rmtree(path)
