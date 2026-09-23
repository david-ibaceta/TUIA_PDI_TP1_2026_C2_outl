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

    
def contar_palabras(caracteres, umbral_espacio=6):
    """
    Función para contar palabras en una lista de caracteres detectados.
    Parámetros:
    - caracteres: lista de diccionarios con información de cada carácter detectado
    - umbral_espacio: umbral para considerar un espacio como separador de palabras
    Retorna:
    - cantidad_palabras: número de palabras detectadas
    """

    # Umbral empírico: un espacio mayor a 6 píxeles suele indicar separación de palabras
    # Si la celda está vacía, no hay palabras
    if not caracteres:
        #print("Cantidad de palabras detectadas: 0 (Celda vacía)")
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

hojas = ["1", "2", "3", "4"]
registros = []

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

    # Itero sobre cada línea horizontal de cada hoja para extraer las celdas correspondientes a Legajo, Nombre, Parcial 1,
    # Parcial 2, Parcial 3 y Condición Final
    print(f"Procesando hoja {hoja}...")
    for i in range(1, len(linea_horizontal) - 1):
        linea_superior = linea_horizontal[i] + 2
        linea_inferior = linea_horizontal[i + 1] - 2
        crop_legajo = img[linea_superior:linea_inferior, linea_vertical[1]+2:linea_vertical[2]-2]
        crop_nombre = img[linea_superior:linea_inferior, linea_vertical[2]+2:linea_vertical[3]-2]
        crop_parcial_1 = img[linea_superior:linea_inferior, linea_vertical[3]+2:linea_vertical[4]-2]
        crop_parcial_2 = img[linea_superior:linea_inferior, linea_vertical[4]+2:linea_vertical[5]-2]
        crop_parcial_3 = img[linea_superior:linea_inferior, linea_vertical[5]+2:linea_vertical[6]-2]
        crop_condicion_final = img[linea_superior:linea_inferior, linea_vertical[6]+2:linea_vertical[7]-2]

        print(f"Registro {i}:")
        #Verificar campo Legajo ==============================================================
        _, cantidad_legajo = buscar_letras(crop_legajo, umbral=138, area_minima=2)
        if cantidad_legajo != 8:
            campo_legajo = "MAL"
        else:
            campo_legajo = "OK"
        print(f"Legajo: {campo_legajo}")
        
        #Verificar campo Nombre y apellido ======================================================
        caracteres_nombre, cantidad_letras = buscar_letras(
            crop_nombre, umbral=138, area_minima=6
        )
        cantidad_palabras = contar_palabras(caracteres_nombre, umbral_espacio=2)

        #Debe contener un minimo de 2 palabras (Apellido y Nombre) y no más de 12 caracteres
        if cantidad_palabras < 2 or cantidad_letras > 12:
            campo_nombre = "MAL"
        else:
            campo_nombre = "OK"
        print(f"Nombre: {campo_nombre}")
      
        
        #Verificar campo Parcial_1 ==============================================================
        caracteres_parcial_1, cantidad_letras = buscar_letras(
            crop_parcial_1, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_parcial_1, umbral_espacio=2)

        # Debe contener 1 o dos caracteres consecutivos (nota del parcial)
        if cantidad_letras < 1 or cantidad_letras > 2:
            campo_parcial_1 = "MAL"
        else:
            campo_parcial_1 = "OK"
        print(f"Parcial 1: {campo_parcial_1}")

        #Verificar campo Parcial_2 ==============================================================
        caracteres_parcial_2, cantidad_letras = buscar_letras(
            crop_parcial_2, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_parcial_2, umbral_espacio=2)

        # Debe contener 1 o dos caracteres consecutivos (nota del parcial)
        if cantidad_letras < 1 or cantidad_letras > 2:
            campo_parcial_2 = "MAL"
        else:
            campo_parcial_2 = "OK"
        print(f"Parcial 2: {campo_parcial_2}")
      
        #Verificar campo Parcial_3 ==============================================================
        caracteres_parcial_3, cantidad_letras = buscar_letras(
            crop_parcial_3, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_parcial_3, umbral_espacio=2)

        # Debe contener 1 o dos caracteres consecutivos (nota del parcial)
        if cantidad_letras < 1 or cantidad_letras > 2:
            campo_parcial_3 = "MAL"
        else:
            campo_parcial_3 = "OK"
        print(f"Parcial 3: {campo_parcial_3}")

        #Verificar campo condicion final ==============================================================
        caracteres_condicion_final, cantidad_letras = buscar_letras(
            crop_condicion_final, umbral=138, area_minima=9
        )
        cantidad_palabras = contar_palabras(caracteres_condicion_final, umbral_espacio=2)

        # Debe contener 1 unica letra (A, R, L)
        if cantidad_letras != 1:
            campo_condicion_final = "MAL"
        else:
            campo_condicion_final = "OK"
        print(f"Condición Final: {campo_condicion_final}")

        registros.append({
            "Hoja": hoja,
            "Id": i,
            "Legajo": campo_legajo,
            "Nombre_apellido": campo_nombre,
            "Parcial_1": campo_parcial_1,
            "Parcial_2": campo_parcial_2,
            "Parcial_3": campo_parcial_3,
            "Condicion_final": campo_condicion_final,
        })


df_resultados = pd.DataFrame(
    registros,
    columns=[
        "Hoja",
        "Id",
        "Legajo",
        "Nombre_apellido",
        "Parcial_1",
        "Parcial_2",
        "Parcial_3",
        "Condicion_final",
    ],
)
df_resultados.to_csv(
    "./git/TUIA_PDI_TP1_2026_C2_outl/TP1/outputs/validacion_resultados.csv",
    index=False,
)
print("Resultados guardados en validacion_resultados.csv")

        

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