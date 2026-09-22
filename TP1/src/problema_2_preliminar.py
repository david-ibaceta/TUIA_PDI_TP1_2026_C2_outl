import cv2
import numpy as np
import matplotlib.pyplot as plt

def buscar_letras(crop_celda, umbral=170, area_minima=2):
    # 1. Convertir a escala de grises si es necesario
    if len(crop_celda.shape) == 3:
        crop = cv2.cvtColor(crop_celda, cv2.COLOR_BGR2GRAY)
    else:
        crop = crop_celda

    # 2. Aplicar el umbral (Binarización)
    _, img_binaria = cv2.threshold(crop, umbral, 255, cv2.THRESH_BINARY_INV)

    # 3. Obtener componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_binaria, connectivity=8)

    # 4. Filtrar por área mínima    
    # Empezamos en 1 para omitir el fondo (label 0)
    caracteres = []
    cantidad = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_minima:
            cantidad += 1
            caracteres.append({
                "componente": i,
                "bbox_xywh": (stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]), # x, y, w, h
                "area": area,
                "centroide": centroids[i] #(cx,cy)
            })
    return caracteres, cantidad

    
def contar_palabras(caracteres, umbral_espacio=6):
    # Umbral empírico: un espacio mayor a 12 píxeles suele indicar separación de palabras
    # Si la celda está vacía, no hay palabras
    if not caracteres:
        print("Cantidad de palabras detectadas: 0 (Celda vacía)")
        return 0

    # --- ORDENAR LOS CARACTERES DE IZQUIERDA A DERECHA ---
    # Ordenamos la lista de diccionarios basándonos en la coordenada X (bbox_xywh[0])
    caracteres_ordenados = sorted(caracteres, key=lambda c: c["bbox_xywh"][0])

    # --- EXTRACCIÓN DE DATOS ---
    # Convertimos las coordenadas X y los Anchos (W) a arreglos de NumPy
    coordenadas_x = np.array([c["bbox_xywh"][0] for c in caracteres_ordenados])
    anchos = np.array([c["bbox_xywh"][2] for c in caracteres_ordenados])

    # --- CALCULAR LOS ESPACIOS ENTRE CARACTERES ---
    # El final de cada carácter es su X + su Ancho
    finales_caracteres = coordenadas_x[:-1] + anchos[:-1]
    # El inicio del siguiente carácter simplemente empieza desde el segundo elemento
    inicios_siguientes = coordenadas_x[1:]
    
    # Restamos los arreglos de NumPy vectorialmente para obtener todos los espacios en un solo paso
    espacios = inicios_siguientes - finales_caracteres

    # --- CONTAR PALABRAS ---
    # Contamos cuántos espacios superan el umbral utilizando NumPy
    espacios_grandes = np.sum(espacios > umbral_espacio)
    
    # La cantidad de palabras siempre es la cantidad de espacios grandes + 1
    cantidad_palabras = espacios_grandes + 1

    return cantidad_palabras


#========================================================================================
#Comienzo del código para procesar las hojas de calificaciones
#========================================================================================

#hojas = ["1", "2", "3", "4"]
hojas = ["4"]

#Itero sobre cada hoja de calificaciones
for hoja in hojas:
    img = cv2.imread(
        f"./git/TUIA_PDI_TP1_2026_C2_outl/TP1/source/grade_sheet_{hoja}.png",
        cv2.IMREAD_GRAYSCALE
    )

    # Determino la posición de las líneas horizontales y verticales para segmentar la hoja en celdas
    # Umbral para binarizar la imagen y detectar líneas
    th = 10
    img_th = img < th
    img_rows = np.sum(img_th, axis=1)
    img_cols = np.sum(img_th, axis=0)

    # Identifico las posiciones de las líneas horizontales y verticales
    (linea_horizontal, ) = np.where(img_rows >= (img_rows.max() - 2))
    alto_columna = linea_horizontal[-1] - linea_horizontal[1]
    (linea_vertical, ) = np.where(img_cols > alto_columna)

    # Itero sobre cada línea horizontal para extraer las celdas correspondientes a Legajo, Nombre, Parcial 1,
    # Parcial 2, Parcial 3 y Condición Final
    for i in range(1, len(linea_horizontal) - 1):
        linea_superior = linea_horizontal[i] + 2
        linea_inferior = linea_horizontal[i + 1] - 2
        crop_legajo = img[linea_superior:linea_inferior, linea_vertical[1]+2:linea_vertical[2]-2]
        crop_nombre = img[linea_superior:linea_inferior, linea_vertical[2]+2:linea_vertical[3]-2]
        crop_parcial_1 = img[linea_superior:linea_inferior, linea_vertical[3]+2:linea_vertical[4]-2]
        crop_parcial_2 = img[linea_superior:linea_inferior, linea_vertical[4]+2:linea_vertical[5]-2]
        crop_parcial_3 = img[linea_superior:linea_inferior, linea_vertical[5]+2:linea_vertical[6]-2]
        crop_condicion_final = img[linea_superior:linea_inferior, linea_vertical[6]+2:linea_vertical[7]-2]

        """
        #Verificar campo Legajo ==============================================================
        caracteres, cantidad = buscar_letras(crop_legajo, umbral=138, area_minima=2)

        print(f"Legajo {i}: {cantidad} letras detectadas.")
        plt.imshow(crop_legajo, cmap="gray")
        plt.title(f"Hoja {hoja} - Legajo {i}: {cantidad} letras detectadas.")
        plt.show()
        
        
        #Verificar campo Nombre ==============================================================
        caracteres_nombre, cantidad_letras = buscar_letras(
            crop_nombre, umbral=138, area_minima=6
        )
        cantidad_palabras = contar_palabras(caracteres_nombre, umbral_espacio=2)

        print(f"Legajo {i}: {cantidad_palabras} palabras detectadas.")
        plt.imshow(crop_nombre, cmap="gray")
        plt.title(f"Hoja {hoja} - Nombre {i}: {cantidad_letras} letras detectadas y {cantidad_palabras} palabras detectadas.")
        plt.show()
        
        
        #Verificar campo Parcial_1 ==============================================================
        caracteres_parcial_1, cantidad_letras = buscar_letras(
            crop_parcial_1, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_parcial_1, umbral_espacio=2)

        print(f"Legajo {i}: {cantidad_palabras} palabras detectadas.")
        plt.imshow(crop_parcial_1, cmap="gray")
        plt.title(f"Hoja {hoja} - Parcial 1 Linea Nro: {i}: {cantidad_letras} letras detectadas.")
        plt.show()
        
        
        #Verificar campo Parcial_2 ==============================================================
        caracteres_parcial_2, cantidad_letras = buscar_letras(
            crop_parcial_2, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_parcial_2, umbral_espacio=2)

        print(f"Legajo {i}: {cantidad_palabras} palabras detectadas.")
        plt.imshow(crop_parcial_2, cmap="gray")
        plt.title(f"Hoja {hoja} - Parcial 2 Linea Nro: {i}: {cantidad_letras} letras detectadas.")
        plt.show()
        
        
        #Verificar campo Parcial_3 ==============================================================
        caracteres_parcial_3, cantidad_letras = buscar_letras(
            crop_parcial_3, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_parcial_3, umbral_espacio=2)

        print(f"Legajo {i}: {cantidad_palabras} palabras detectadas.")
        plt.imshow(crop_parcial_3, cmap="gray")
        plt.title(f"Hoja {hoja} - Parcial 3 Linea Nro: {i}: {cantidad_letras} letras detectadas.")
        plt.show()

        
        #Verificar campo condicion final ==============================================================
        caracteres_condicion_final, cantidad_letras = buscar_letras(
            crop_condicion_final, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_condicion_final, umbral_espacio=2)

        print(f"Legajo {i}: {cantidad_palabras} palabras detectadas.")
        plt.imshow(crop_condicion_final, cmap="gray")
        plt.title(f"Hoja {hoja} - Condición Final Linea Nro: {i}: {cantidad_letras} letras detectadas.")
        plt.show()

        """

"""
def main():
    # 1. Simulación de la carga y llamada a tu función modificada
    crop = cv2.imread("celda_crop.png") 

    caracteres, cantidad = buscar_letras(crop)

    if cantidad > 12:
        print("Validación fallida: Se esperaba menos de 12 caracteres.")
    # Falta validaciones!!!!!!
    cantidad_palabras = contar_palabras(caracteres, umbral_espacio=3)

    # --- VALIDACIÓN DE EJEMPLO (Nombre y Apellido) ---
    print(f"Caracteres válidos detectados: {len(caracteres)}")
    print(f"Cantidad de palabras detectadas: {cantidad_palabras}")
    
    if cantidad_palabras >= 2:
        print("Validación exitosa: Contiene al menos dos palabras (posible Apellido y Nombre).")
    else:
        print("Validación fallida: Se esperaba más de una palabra.")


if __name__ == "__main__":
    main()

    """