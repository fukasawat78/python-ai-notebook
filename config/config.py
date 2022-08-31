import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Path settings
HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONF_DIR = os.path.join(HOME, "config")
DATA_DIR = os.path.join(HOME, "data")

# seed
SEED = 1234