"""Utility helpers.

This module provides helper functions for saving Python objects
and other common utilities used across the training and prediction pipelines.
"""

import os
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException

import joblib


def save_object(obj, file_path):
    """Serialize an object to disk using joblib.

    Parameters
    ----------
    obj : any
        The Python object to serialize.
    file_path : str
        Path to the target file where the object will be saved.

    Raises
    ------
    CustomException
        If saving fails due to an I/O or serialization error.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Deserialize and load a saved object from disk.

    Parameters
    ----------
    file_path : str
        Path to the file containing the serialized object.

    Returns
    -------
    any
        The deserialized Python object.

    Raises
    ------
    CustomException
        If loading fails due to an I/O or deserialization error.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_config():
    """Load the YAML configuration file.

    Returns:
        dict: Parsed configuration values from configs/config.yaml.
    """
    config_path = Path("configs") / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
