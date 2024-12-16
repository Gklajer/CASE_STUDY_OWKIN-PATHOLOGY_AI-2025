import sys

from .utils.constants import (
    KAGGLE_HELPER_FUNCTIONS_HANDLE,
    KAGGLE_INPUT_HELPER_FUNCTION_PATH,
)
from .utils.kaggle import download_from_kaggle_if_missing

HELPER_FUNCTIONS_PATH = download_from_kaggle_if_missing(
    KAGGLE_INPUT_HELPER_FUNCTION_PATH, KAGGLE_HELPER_FUNCTIONS_HANDLE
)
sys.path.append(HELPER_FUNCTIONS_PATH)
