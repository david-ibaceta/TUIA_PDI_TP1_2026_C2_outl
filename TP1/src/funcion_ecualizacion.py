import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ecualizacion_local(img, M, N):
    """
    Realiza la ecualización local de histograma.

    Parámetros:
        img : imagen en escala de grises (uint8)
        M   : cantidad de filas de la ventana
        N   : cantidad de columnas de la ventana

    Retorna:
        Imagen ecualizada localmente.
    """

    # 1. Verificamos que la imagen sea de escala de grises
    
    if len(img.shape) != 2:
        raise ValueError("La imagen debe estar en escala de grises")

    # 2. Verificamos que los tamaños de ventana sean positivos
    
    if M <= 0 or N <= 0:
        raise ValueError("M y N deben ser mayores que cero")


    # 3. Calculamos cuánto debemos desplazarnos desde el
    #    centro de la ventana hacia cada borde.
    
    mitad_M = M // 2
    mitad_N = N // 2


    # 4. Agregamos un borde a la imagen.
    
    img_borde = cv2.copyMakeBorder(
        img,
        mitad_M,
        mitad_M,
        mitad_N,
        mitad_N,
        cv2.BORDER_REPLICATE
    )

    
    # 5. Creamos la imagen de salida.

    salida = np.zeros_like(img)

    # 6. Recorremos todos los píxeles de la imagen original.

    filas, columnas = img.shape

    for i in range(filas):

        for j in range(columnas):

            # 7. Extraemos la ventana correspondiente al píxel (i, j).
            ventana = img_borde[
                i:i + M,
                j:j + N
            ]

            # 8. Calculamos el histograma de la ventana.
            histograma = np.bincount(
                ventana.ravel(),
                minlength=256
            )
            # 9. Calculamos la distribución acumulada (CDF).

            cdf = np.cumsum(histograma)

            # 10. Obtenemos el valor mínimo de la CDF que
            #     corresponde a un nivel de intensidad presente
            #     en la ventana.

            cdf_min = cdf[histograma > 0].min()

            # 11. Obtenemos el valor del píxel central.
            valor_pixel = ventana[mitad_M, mitad_N]

            # Si todos los píxeles de la ventana tienen el mismo valor,
            # no hay contraste local para ecualizar.
            if cdf_min == M * N:
                nuevo_valor = valor_pixel

            else:
                # 12. Aplicamos la transformación de ecualización
                #     al píxel central.
    
                nuevo_valor = (
                    (cdf[valor_pixel] - cdf_min)
                    /
                    (M * N - cdf_min)
                    * 255
                )

            # 13. Guardamos el nuevo valor en la imagen de salida.

            salida[i, j] = np.clip(
                nuevo_valor,
                0,
                255
            )

    return salida


# -------------------------------------------------------------
# PROGRAMA PRINCIPAL
# -------------------------------------------------------------

# Cargamos la imagen en escala de grises
img = cv2.imread(
    "./git/TUIA_PDI_TP1_2026_C2_outl/TP1/source/Imagen_con_detalles_escondidos.tif",
    cv2.IMREAD_GRAYSCALE
)

# Verificamos que la imagen se haya cargado correctamente
if img is None:
    raise FileNotFoundError(
        "No se pudo encontrar la imagen"
    )


# Aplicamos la ecualización local utilizando una ventana
# de píxeles impares.
resultado_3x3 = ecualizacion_local(img, 3, 3)
resultado_5x5 = ecualizacion_local(img, 5, 5)
resultado_7x7 = ecualizacion_local(img, 7, 7)
resultado_15x15 = ecualizacion_local(img, 15, 15)

# Guardamos todas las imágenes para mostrarlas juntas con Matplotlib.
imagenes = [
    img,
    resultado_3x3,
    resultado_5x5,
    resultado_7x7,
    resultado_15x15,
]
titulos = [
    "Imagen original",
    "Ecualizacion local 3x3",
    "Ecualizacion local 5x5",
    "Ecualizacion local 7x7",
    "Ecualizacion local 15x15",
]

figura, ejes = plt.subplots(1, len(imagenes), figsize=(20, 5))

for eje, imagen, titulo in zip(ejes, imagenes, titulos):
    eje.imshow(imagen, cmap="gray", vmin=0, vmax=255)
    eje.set_title(titulo)
    eje.axis("off")

figura.tight_layout()
plt.show()