"""
Filesystem helpers: collect JPGs, find templates, and next defect index.
"""
import os
import re
from typing import List, Tuple

def gather_jpgs_recursive(root: str) -> List[str]:
    """
    Recursively collect .jpg/.jpeg files under root.
    """
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fl = fn.lower()
            if fl.endswith(".jpg") or fl.endswith(".jpeg"):
                out.append(os.path.join(dirpath, fn))
    return out

def ensure_template_assets(root_with_label: str, class_name: str) -> Tuple[str, str]:
    """
    Returns (template_image_path, template_masks_dir). Raises if missing.
    """
    tmpl_img = os.path.join(root_with_label, class_name, f"{class_name}.png")
    tmpl_masks = os.path.join(root_with_label, class_name, "masks")
    if not os.path.isfile(tmpl_img):
        raise FileNotFoundError(f"Template image not found: {tmpl_img}")
    if not os.path.isdir(tmpl_masks):
        raise FileNotFoundError(f"Template masks folder not found: {tmpl_masks}")
    return tmpl_img, tmpl_masks

def next_defect_index(class_defect_dir: str, class_name: str) -> int:
    """
    Find next index for subfolders named f\"{class_name}_defect_<n>\" under class_defect_dir.
    """
    if not os.path.isdir(class_defect_dir):
        return 1
    pat = re.compile(rf"^{re.escape(class_name)}_defect_(\d+)$")
    max_idx = 0
    for e in os.scandir(class_defect_dir):
        if e.is_dir():
            m = pat.match(e.name)
            if m:
                try:
                    max_idx = max(max_idx, int(m.group(1)))
                except ValueError:
                    pass
    return max_idx + 1