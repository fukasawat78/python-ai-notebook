import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Constants:
    
    HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    CONF_DIR = os.path.join(HOME, "config")
    DATA_DIR = os.path.join(HOME, "data")
    DATASET_PATH = DATA_DIR + "/01_mobile_price_classification"

    # seed
    SEED = 1234