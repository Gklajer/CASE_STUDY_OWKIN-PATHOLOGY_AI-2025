import os
from typing import Union

import kagglehub

from .constants import (
    KAGGLE_DATA_HANDLE,
    KAGGLE_INPUT_DATA_PATH,
    KAGGLE_WEIGHTS_HANDLE,
    KAGGLE_WEIGHTS_PATH,
)


def download_from_kaggle_if_missing(
    target_path: Union[os.PathLike, str],
    kaggle_handle: Union[os.PathLike, str],
    mode: str = "dataset",
) -> os.PathLike:
    """
    Download the file(s) from Kaggle if missing.

    Args:
        target_path: The path to the target file(s).
        kaggle_handle: The Kaggle handle of the dataset or model.
        mode: The mode 'dataset' | 'model'.

    Returns:
        The path to the target file(s).
    """

    if os.path.exists(target_path):
        return target_path

    download_func = (
        kagglehub.dataset_download if mode == "dataset" else kagglehub.model_download
    )

    return os.path.join(
        download_func(kaggle_handle),
        os.path.basename(target_path),
    )


def fetch_kaggle_paths() -> os.PathLike:
    """
    Fetch the model weights, data from Kaggle.

    Returns:
        The path to the model weights, data.
    """

    weights_path = download_from_kaggle_if_missing(
        KAGGLE_WEIGHTS_PATH, KAGGLE_WEIGHTS_HANDLE, mode="model"
    )
    data_path = download_from_kaggle_if_missing(
        KAGGLE_INPUT_DATA_PATH, KAGGLE_DATA_HANDLE
    )

    return weights_path, data_path
