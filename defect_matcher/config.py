"""
Config: dataset mapping and default parameters for defect matching pipeline.
"""
from typing import Dict

# Map defect dataset folder name -> circuit class
DATASET_TO_CLASS: Dict[str, str] = {
    # 27QHD
    "AA-270QHD-0219": "27QHD",
    "TL270AK2BA01": "27QHD",
    # 238FHD
    "238FHD-TSRTP": "238FHD",
    "AA_238FHD": "238FHD",
    "AA-238FHD-0321": "238FHD",
    "AA_238FHD_1": "238FHD",
    # 270FHD
    "270FHD-TOITP": "270FHD",
    "AA_270FHD": "270FHD",
    "AA-270FHD-0510": "270FHD",
    # TL156
    "AA_156DLS_0424": "TL156",
    "AA-156DLS-0423": "TL156",
    # 238QHD
    "AA_238QHD": "238QHD",
    "TL238AE2BA01": "238QHD",
    "AA-238QHD-0321": "238QHD",
    # 245FHD
    "AA_245FHD_0414": "245FHD",
    "TL245A12BA01": "245FHD",
    # 215FHD
    "TL215A3BBA01": "215FHD",
}

DEFAULT_SOURCE_ROOT = "/mnt/workspace/autorepair_t9/data/最终通用模型/full_train_data"

# Defaults for CLI options
DEFAULT_NORMALIZE_SIZE = 512
DEFAULT_ROTATIONS = [0, 90, 180, 270]     # degrees clockwise to try
DEFAULT_MIN_SCALE = 0.85
DEFAULT_MAX_SCALE = 1.15
DEFAULT_SCALE_STEPS = 13
DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_SAVE_MATCHED_VIS = False
DEFAULT_LIMIT_PER_DATASET = 20             # 0 means no cap
DEFAULT_RANDOM_ROTATE_SAMPLE = True       # random 0/90/180/270 applied to aug_image + masks