from functools import partial, update_wrapper
from os import PathLike
from typing import Union

import numpy as np
from PIL import Image


def wrapped_partial(func, *args, **kwargs) -> partial:
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def get_raw_image(tiles_path: Union[str, PathLike], roi_name: str) -> np.ndarray:
    """Get the raw image.

    Args:
        tiles_path : The path to the folder containing the tiles saved as <roi_name>.png.
        roi_name: The name of the region of interest.

    Returns:
        The raw image.
    """

    raw_image_path = tiles_path / (roi_name + ".png")

    with open(raw_image_path, "rb") as fin:
        pillow_raw_image = Image.open(fin)
        pillow_raw_image = pillow_raw_image.convert("RGB")
        raw_image = np.asarray(pillow_raw_image, dtype=np.uint8)

    return raw_image[:, :, :3]
