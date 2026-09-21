import cv2
import numpy as np
import matplotlib.pyplot as plt

#hojas = ["1", "2", "3", "4"]
hojas = ["1"]

for hoja in hojas:
    img = cv2.imread(
        f"./git/TUIA_PDI_TP1_2026_C2_outl/TP1/source/grade_sheet_{hoja}.png",
        cv2.IMREAD_GRAYSCALE
    )

    th = 10
    img_th = img < th
    img_rows = np.sum(img_th, axis=1)
    img_cols = np.sum(img_th, axis=0)

    (linea_horizontal, ) = np.where(img_rows >= (img_rows.max() - 2))
    alto_columna = linea_horizontal[-1] - linea_horizontal[1]
    (linea_vertical, ) = np.where(img_cols > alto_columna)
    #print("Alto de la columna:", alto_columna)
    
    
    print("#" * 70)
    print("Hoja:", hoja)
    print("Renglones:", len(linea_horizontal), "en posiciones:", linea_horizontal)
    print("Columnas:", len(linea_vertical), "en posiciones:", linea_vertical)
    
    for i in range(len(linea_horizontal) - 2):
        img_legajo = img[linea_horizontal[i + 1]+2:linea_horizontal[i + 2]-2, linea_vertical[1]+2:linea_vertical[2]-2]
        mascara_legajo = np.uint8(img_legajo < 170) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_legajo, connectivity=8, ltype=cv2.CV_32S)
        area_minima = 2
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        componentes_validas = np.sum(areas >= area_minima)
        print(f"Legajo {i + 1}: {componentes_validas} componentes conectadas")
        plt.imshow(mascara_legajo, cmap="gray")
        plt.title(f"Hoja {hoja} - Legajo {i + 1}")
        plt.show()

