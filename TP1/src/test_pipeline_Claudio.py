import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
source_dir = base_dir / "source"

for idx in [1, 2, 3, 4]:
    img = cv2.imread(str(source_dir / f"grade_sheet_{idx}.png"), cv2.IMREAD_GRAYSCALE)
    img_th = (img < 128).astype(np.uint8)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    v_lines = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel_v)
    col_sum = np.sum(v_lines, axis=0)
    
    th = 0.4 * np.max(col_sum)
    idx_peaks = np.where(col_sum > th)[0]
    groups = np.split(idx_peaks, np.where(np.diff(idx_peaks) > 1)[0] + 1)
    x_lines = [(g[0] + g[-1]) // 2 for g in groups]
    print(f"Sheet {idx}: x_lines={len(x_lines)}, col diffs = {np.diff(x_lines)}")

