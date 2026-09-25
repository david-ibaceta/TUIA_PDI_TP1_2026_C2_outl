import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def buscar_letras(crop_celda, umbral=170, area_minima=2):
    """
    Función para detectar caracteres en una celda de una hoja de calificaciones.
    Parámetros:
    - crop_celda: imagen recortada de la celda
    - umbral: umbral para la binarización
    - area_minima: área mínima para considerar un componente como un carácter
    Retorna:
    - caracteres: lista de diccionarios con información de cada carácter detectado
    - cantidad: número de caracteres detectados
    """

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

def clasificar_letra(imagen_crop, umbral_ruido = 10, umbral_simetria = 50):
    # 1. Convertir a escala de grises si no lo está
    if len(imagen_crop.shape) == 3:
        gris = cv2.cvtColor(imagen_crop, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen_crop.copy()

    # 2. Binarizar (Invertir para que la letra sea BLANCA (255) y el fondo NEGRO (0))
    # Esto facilita contar píxeles de la letra usando np.sum() o cv2.countNonZero
    _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. Redimensionar a un tamaño estándar (ej. 40x40) para que las proporciones siempre coincidan
    letra = cv2.resize(binaria, (40, 40))

    # 4. Dividir la letra en regiones usando SLICING
    # [filas, columnas] -> alto 40, ancho 40
    mitad_superior_derecha = letra[0:20, 20:40]
    mitad_inferior_derecha = letra[20:40, 20:40]
    lado_izquierdo         = letra[:, 0:20]
    lado_derecho           = letra[:, 20:40]

    # 5. Contar cuántos píxeles "de letra" hay en cada zona
    pixels_sup_der = np.sum(mitad_superior_derecha == 255)
    pixels_inf_der = np.sum(mitad_inferior_derecha == 255)

    # Calcular simetría izquierda-derecha (Restamos las matrices de ambos lados)
    # Si es muy simétrica (como la 'A'), la diferencia absoluta será baja
    diferencia_simetria = np.sum(cv2.absdiff(lado_izquierdo, cv2.flip(lado_derecho, 1)) == 255)

    # 6. Lógica de decisión basada en umbrales (puedes ajustar estos números según tus pruebas)
    # Una 'L' no tiene casi nada en la esquina superior derecha
    if pixels_sup_der < umbral_ruido: # Umbral de ruido a ajustar
        return "L"

    # Una 'A' es altamente simétrica comparada con una 'R'
    # y la 'R' tiene una "panza" arriba a la derecha y una "pata" abajo a la derecha
    if diferencia_simetria < umbral_simetria: # Umbral de simetría a ajustar
        return "A"
    else:
        return "R"

def contar_palabras(caracteres, umbral_espacio=3):
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
hojas = ["2"]

#Itero sobre cada hoja de calificaciones

for hoja in hojas:
    img = cv2.imread(
        f"./git/TUIA_PDI_TP1_2026_C2_outl/TP1/source/grade_sheet_{hoja}.png",
        cv2.IMREAD_GRAYSCALE
    )
    
    # Binarizamos la imagen para detectar las líneas oscuras de la grilla
    img_binaria = img < 10

    # Determino la posición de las líneas horizontales y verticales para segmentar la hoja en celdas
    img_rows = np.sum(img_binaria, axis=1)
    img_cols = np.sum(img_binaria, axis=0)

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

        #Verificar campo Legajo ==============================================================
        caracteres, cantidad = buscar_letras(crop_legajo, umbral=138, area_minima=3)

        print(f"Legajo {i}: {cantidad} letras detectadas.")
        plt.imshow(crop_legajo, cmap="gray")
        plt.title(f"Hoja {hoja} - Legajo {i}: {cantidad} letras detectadas.")
        plt.show()

        """

        #Verificar campo Nombre ==============================================================
        caracteres_nombre, cantidad_letras = buscar_letras(
            crop_nombre, umbral=138, area_minima=6
        )
        cantidad_palabras = contar_palabras(caracteres_nombre, umbral_espacio=6)

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
        if caracteres_condicion_final:
            c = caracteres_condicion_final[0]["bbox_xywh"]
            crop_letra = crop_condicion_final[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]
            calificacion = clasificar_letra(crop_letra)
        else:
            calificacion = "No detectada"

        # Debe contener 1 unica letra (A, R, L)
        if cantidad_letras != 1:
            campo_condicion_final = "MAL"
            calificacion = "No detectada"
        else:
            campo_condicion_final = "OK"

        print(f"Condición Final: {campo_condicion_final} ({calificacion})")
        print(f"Legajo {i}: Calificación detectada: {calificacion}")

        print(f"Legajo {i}: {cantidad_palabras} palabras detectadas.")
        plt.imshow(crop_condicion_final, cmap="gray")
        plt.title(f"Hoja {hoja} - Condición Final Linea Nro: {i}: Calificación detectada: {calificacion}")
        plt.show()

        """