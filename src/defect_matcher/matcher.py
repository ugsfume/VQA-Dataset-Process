"""
Template matcher: scale sweep, rotation sweep, and visualization.
"""
from typing import Dict, List
import cv2
import numpy as np

from image_ops import rotate_image_90

def build_scales(min_scale: float, max_scale: float, steps: int) -> List[float]:
    steps = max(2, int(steps))
    arr = np.linspace(min_scale, max_scale, steps)
    return [round(float(s), 3) for s in arr]

def multi_scale_match(full_bgr, template_bgr, scales: List[float], method: int = cv2.TM_CCOEFF_NORMED) -> Dict:
    """
    Grayscale multi-scale matching. Returns best match dict with score and bbox.
    """
    full_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    H, W = full_gray.shape
    th0, tw0 = tmpl_gray.shape
    best = {
        "score": -1.0, "scale": None, "x1": 0, "y1": 0, "x2": 0, "y2": 0,
        "w": 0, "h": 0, "method": "TM_CCOEFF_NORMED"
    }
    for s in scales:
        tw = int(round(tw0 * s))
        th = int(round(th0 * s))
        if tw < 8 or th < 8 or tw > W or th > H:
            continue
        tmpl_s = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        res = cv2.matchTemplate(full_gray, tmpl_s, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        score = float(max_val)
        if score > best["score"]:
            tlx, tly = max_loc
            best.update({
                "score": score, "scale": float(s),
                "x1": int(tlx), "y1": int(tly),
                "w": int(tw), "h": int(th),
                "x2": int(tlx + tw - 1), "y2": int(tly + th - 1)
            })
    return best

def find_best_match_with_rotations(full_bgr, template_bgr, rotations: List[int], scales: List[float]) -> Dict:
    """
    Try rotations (deg CW) and keep the best-scoring match. Adds 'rot_deg' to result.
    """
    best_overall = None
    for rot in rotations:
        tmpl_rot = rotate_image_90(template_bgr, rot)
        match = multi_scale_match(full_bgr, tmpl_rot, scales=scales)
        match["rot_deg"] = int(rot)
        if best_overall is None or match["score"] > best_overall["score"]:
            best_overall = match
    return best_overall

def draw_match(full_bgr, match: Dict, out_path: str, rot_deg: int = 0) -> None:
    """
    Save visualization of matched rectangle on the full image.
    """
    vis = full_bgr.copy()
    x1, y1, x2, y2 = match["x1"], match["y1"], match["x2"], match["y2"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 225, 0), thickness=3)
    label = f"score={match['score']:.3f} scale={match['scale']:.3f} rot={rot_deg}"
    cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)