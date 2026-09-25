import cv2
import numpy as np
from problema2 import detectar_lineas_grilla

img = cv2.imread("TP1/source/grade_sheet_1.png", cv2.IMREAD_GRAYSCALE)
y_filas, x_cols = detectar_lineas_grilla(img)

for r in [6, 7, 8, 12]:
    celda = img[y_filas[r]:y_filas[r+1], x_cols[1]:x_cols[2]]
    interior = celda[2:-2, 2:-2]
    b = (interior < 120).astype(np.uint8)
    col_sum = np.sum(b, axis=0)
    print(f"\n--- Row {r+1} ---")
    print("Col sum:", col_sum[col_sum > 0])

