import os
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
import torch.nn.functional as F

from transformers.optimization import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

from owkin_case_study.postprocess.instance_map import generate_instance_map
from owkin_case_study.viz.visualize import get_image_with_masks

from .constants import VIRIDIS_RGB_LIST


def _masks_to_instance_map(masks: np.ndarray) -> np.ndarray:
    """Convert masks to instance map.

    Args:
        masks: The masks.

    Returns:
        The instance map.
    """

    return np.concatenate(
        (
            (masks == 0).all(axis=0, keepdims=True),
            masks,
        ),
        axis=0,
    ).argmax(axis=0)


def _extract_instance_map_and_labels(
    np_branch: np.ndarray, tp_branch: np.ndarray, hv_branch: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Extract the instance map and labels.

    Args:
        np_branch: The nuclear probability branch.
        tp_branch: The type branch.
        hv_branch: The horizontal-vertical distance to nuclei branch.

    Returns:
        The instance map and labels.
    """

    seg_dict = generate_instance_map(np_branch, tp_branch, hv_branch)

    inst_map = (
        _masks_to_instance_map(seg_dict["masks"])
        if seg_dict["masks"].size > 0
        else np.zeros_like(np_branch[0])
    )

    return inst_map, seg_dict["labels"]


def _process_segmentation(
    tiles_path: str,
    np_branch: Tensor,
    tp_branch: Tensor,
    hv_branch: Tensor,
    roi_name: str,
    rgb_list: list[tuple[int, int, int]] = VIRIDIS_RGB_LIST,
) -> np.ndarray:
    inst_map, labels = _extract_instance_map_and_labels(np_branch, tp_branch, hv_branch)

    _, seg_img = get_image_with_masks(
        tiles_path, roi_name, inst_map, labels, cmap=rgb_list, line_thickness=2
    )

    return seg_img


def extract_mask_with_instance_labels(
    np_branch: np.ndarray, tp_branch: np.ndarray, hv_branch: np.ndarray
) -> np.ndarray:
    """Extract the mask with instance labels.

    Args:
        np_branch: The nuclear probability branch.
        tp_branch: The type branch.
        hv_branch: The horizontal-vertical distance to nuclei branch.

    Returns:
        The mask with instance labels.
    """

    inst_map, labels = _extract_instance_map_and_labels(np_branch, tp_branch, hv_branch)

    get_corresponding_labels = np.vectorize(lambda x: 0 if x == 0 else labels[x - 1])

    return get_corresponding_labels(inst_map)


def display_segmentation(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    roi_names: list[str],
    tiles_path: Union[str, os.PathLike],
):
    preds = (F.softmax(outputs["np"], dim=1), outputs["tp"].argmax(1), outputs["hv"])
    targets = (
        F.one_hot(targets["np"].long()).permute(0, 3, 1, 2),
        targets["tp"],
        targets["hv"].permute(0, 3, 1, 2),
    )

    preds, targets = tuple(
        zip(*(list(branch.cpu().detach().numpy()) for branch in branchs))
        for branchs in (
            preds,
            targets,
        )
    )

    for i, (pred, target, roi_name) in enumerate(zip(preds, targets, roi_names)):
        for j, (branchs, title) in enumerate(
            zip((pred, target), ("Predicted", "Target"))
        ):
            seg_img = _process_segmentation(
                tiles_path,
                *branchs,
                roi_name=roi_name,
            )

            plt.subplot(len(roi_names), 2, 2 * i + j + 1)
            plt.imshow(seg_img)
            plt.axis("off")
            plt.title(title)
    plt.show()
