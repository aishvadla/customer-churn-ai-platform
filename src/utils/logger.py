"""Logging configuration for the auto fuel efficiency project.

This module creates a file-based logger that captures informational and error
messages for the project's training and prediction workflows.
"""

import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path.cwd() / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logger = logging.getLogger("fuel_efficiency")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.propagate = False
