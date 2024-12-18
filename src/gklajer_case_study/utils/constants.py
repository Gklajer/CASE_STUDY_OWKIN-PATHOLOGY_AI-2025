import matplotlib.pyplot as plt
import os

NUMBER_CELL_TYPES = 10
DEFAULT_CLASS_LABELS = {i: f"Cell type {i}" for i in range(1, NUMBER_CELL_TYPES)}
VIRIDIS_CMAP = plt.get_cmap("viridis")
VIRIDIS_RGB_LIST = [
    tuple(int(c * 255) for c in VIRIDIS_CMAP(i)[:3]) for i in range(NUMBER_CELL_TYPES)
]


KAGGLE_DATA_HANDLE = os.path.join("rftexas", "tiled-consep-224x224px")

KAGGLE_WEIGHTS_HANDLE = os.path.join("afiltowk", "phikon/PyTorch/default/1")

KAGGLE_INPUT_DATA_PATH = "/kaggle/input/tiled-consep-224x224px/consep"
KAGGLE_WEIGHTS_PATH = "/kaggle/input/phikon/pytorch/default/1/phikon.pth"

KAGGLE_HELPER_FUNCTIONS_HANDLE = os.path.join(
    "rftexas", "owkin-case-study-helper-functions"
)

KAGGLE_INPUT_HELPER_FUNCTION_PATH = "/kaggle/input/owkin-case-study-helper-functions/"
