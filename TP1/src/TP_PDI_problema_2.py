import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import csv

# 1. Encontrar la ruta absoluta del proyecto para poder acceder a las carpetas 'source' y 'outputs'
# __file__ obtiene la posición de main.py. .parent nos saca de 'scr/' y nos deja en la raíz 'TP_PDI'
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Definir las rutas de las carpetas 'source' y 'outputs' usando la ruta base
SOURCE_DIR = BASE_DIR / "source"
OUTPUTS_DIR = BASE_DIR / "outputs"
DEBUG_DIR = BASE_DIR / "debug"

# Asegurar que la carpeta 'outputs' exista en la pc de los colaboradores
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class Record_crops:
    def __init__(self, 
                crop_legajo, 
                crop_nombre, 
                crop_parcial_1, 
                crop_parcial_2, 
                crop_parcial_3, 
                crop_condicion_final):
        self.crop_legajo = crop_legajo
        self.crop_nombre = crop_nombre
        self.crop_parcial_1 = crop_parcial_1
        self.crop_parcial_2 = crop_parcial_2
        self.crop_parcial_3 = crop_parcial_3
        self.crop_condicion_final = crop_condicion_final

    def save( self, img:int, row:int):
        sr = "{:03d}".format(row)
        if not cv2.imwrite(str(DEBUG_DIR / f"crop.{img}.{sr}.1.legajo.png"), self.crop_legajo):
            pass # ignore exceptions for debug
        if not cv2.imwrite(str(DEBUG_DIR / f"crop.{img}.{sr}.2.nombre.png"), self.crop_nombre):
           pass
        if not cv2.imwrite(str(DEBUG_DIR / f"crop.{img}.{sr}.3.parcial_1.png"), self.crop_parcial_1):
            pass
        if not cv2.imwrite(str(DEBUG_DIR / f"crop.{img}.{sr}.4.parcial_2.png"), self.crop_parcial_2):
            pass
        if not cv2.imwrite(str(DEBUG_DIR / f"crop.{img}.{sr}.5.parcial_3.png"), self.crop_parcial_3):
            pass
        if not cv2.imwrite(str(DEBUG_DIR / f"crop.{img}.{sr}.6.condicion_final.png"), self.crop_condicion_final):
            pass

class Record_check:
    def __init__(self,                          
                 campo_legajo,
                 campo_nombre,
                 campo_parcial_1,
                 campo_parcial_2,
                 campo_parcial_3,
                 campo_condicion_final,
                 calificacion ):
        self.campo_legajo = campo_legajo
        self.campo_nombre = campo_nombre
        self.campo_parcial_1 = campo_parcial_1
        self.campo_parcial_2 = campo_parcial_2
        self.campo_parcial_3 = campo_parcial_3
        self.campo_condicion_final =campo_condicion_final
        self.calificacion = calificacion
        self.registro_valido = (self.campo_legajo == "OK" and 
                                self.campo_nombre == "OK" and 
                                self.campo_parcial_1 == "OK" and 
                                self.campo_parcial_2 == "OK" and 
                                self.campo_parcial_3 == "OK" and 
                                self.campo_condicion_final == "OK")

    def report(self, i:int):
        print(f"\nRegistro {i}:{"Válido" if self.registro_valido else "Inválido"}")
        print(f"Legajo: {self.campo_legajo}")
        print(f"Nombre: {self.campo_nombre}")
        print(f"Parcial 1: {self.campo_parcial_1}")
        print(f"Parcial 2: {self.campo_parcial_2}")
        print(f"Parcial 3: {self.campo_parcial_3}")
        print(f"Condición Final: {self.campo_condicion_final} ({self.calificacion})")



class Record:
    def __init__(self, 
                 id:int, 
                 record_crops:Record_crops,
                 record_check:Record_check):
        self.id = id
        self.record_crops = record_crops
        self.record_check = record_check



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
    """
    Función para detectar caracteres en una celda de una hoja de calificaciones.
    Parámetros:
    - imagen_crop: imagen recortada de la celda
    - umbral_ruido: umbral para la detección letra L (ruido)
    - umbral_simetria: umbral para la detección letra A (simetria)
    Retorna:
    - String con caracteres detectados
    """

    # 1. Convertir a escala de grises si no lo está
    if len(imagen_crop.shape) == 3:
        gris = cv2.cvtColor(imagen_crop, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen_crop.copy()

    # 2. Binarizar (Invertir para que la letra sea BLANCA (255) y el fondo NEGRO (0))
    # Esto facilita contar píxeles de la letra usando np.sum()
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

    # 6. Lógica de decisión basada en umbrales (ajustar estos números según las pruebas)
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
    """
    Función para contar palabras en una lista de caracteres detectados.
    Parámetros:
    - caracteres: lista de diccionarios con información de cada carácter detectado
    - umbral_espacio: umbral para considerar un espacio como separador de palabras
    Retorna:
    - cantidad_palabras: número de palabras detectadas
    """

    # Umbral empírico: un espacio mayor suele indicar separación de palabras
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

def generar_imagen(df):
    """
    Función para generar una única imagen de salida con los registros de los
    alumnos no aprobados.
    Parámetros:
    - df: Dataframe con los registros válidos que tienen calificación R Y L
    Retorna:
    - Imágen de salida
    """
    registros = df[
        df["Registro_valido"]
        & df["Condicion_final_letra"].isin(["L", "R"])
    ]
    alto_registro = 160
    ancho_salida = 900
    margen = 20
    alto_salida = max(100, margen + alto_registro * len(registros))
    img_salida = np.full((alto_salida, ancho_salida, 3), 255, dtype=np.uint8)

    for indice, (_, registro) in enumerate(registros.iterrows()):
        y = margen + indice * alto_registro
        hoja = Path(registro["Hoja"])
        file = Path(SOURCE_DIR / f"grade_sheet_{hoja}.png")
        imagen = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            raise FileNotFoundError(f"No se pudo abrir la imagen: {hoja}")

        a, b, c, d = (int(registro[columna]) for columna in ("a", "b", "c", "d"))
        crop = imagen[a:b, c:d]
        if crop.size == 0:
            continue

        alto_crop, ancho_crop = crop.shape
        escala = min(480 / ancho_crop, 100 / alto_crop)
        dimensiones = (max(1, int(ancho_crop * escala)), max(1, int(alto_crop * escala)))
        crop = cv2.resize(crop, dimensiones, interpolation=cv2.INTER_AREA)
        crop_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        y_crop = y + 40 + (100 - crop_color.shape[0]) // 2
        img_salida[y_crop:y_crop + crop_color.shape[0], margen:margen + crop_color.shape[1]] = crop_color

        letra = registro["Condicion_final_letra"]
        observacion = "Libre" if letra == "L" else "Recupera"
        hoja_id = hoja.stem.removeprefix("grade_sheet_")
        cv2.putText(
            img_salida,
            f"Hoja {hoja_id} | Registro {indice + 1}",
            (margen, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_salida,
            f"Calificacion: {letra}",
            (540, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_salida,
            f"Observacion: {observacion}",
            (540, y + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            img_salida,
            (margen, y + alto_registro - 1),
            (ancho_salida - margen, y + alto_registro - 1),
            (200, 200, 200),
            1,
        )

    ruta_salida = OUTPUTS_DIR / "no_aprobados.png"
    if not cv2.imwrite(str(ruta_salida), img_salida):
        raise OSError(f"No se pudo guardar la imagen: {ruta_salida}")
    return img_salida

def verificar_legajo(crop_legajo):
    #Verificar campo Legajo ==============================================================
    _, cantidad_legajo = buscar_letras(crop_legajo, umbral=138, area_minima=3)
    if cantidad_legajo != 8:
        return "MAL"
    else:
        return "OK"

def verificar_nombre(crop_nombre):
    #Verificar campo Nombre y apellido ======================================================
    caracteres_nombre, cantidad_letras = buscar_letras(crop_nombre, umbral=138, area_minima=6)
    cantidad_palabras = contar_palabras(caracteres_nombre, umbral_espacio = 6)
    #Debe contener un minimo de 2 palabras (Apellido y Nombre) y no más de 12 caracteres
    if cantidad_palabras < 2 or cantidad_letras > 12:
        return "MAL"
    else:
        return "OK"

def verificar_parcial(crop_parcial):
    #Verificar campo Parcial_1 ==============================================================
    caracteres_parcial_1, cantidad_letras = buscar_letras(crop_parcial, umbral=138, area_minima=9)
    cantidad_palabras = contar_palabras(caracteres_parcial_1, umbral_espacio=2)
    # Debe contener 1 o dos caracteres consecutivos (nota del parcial)
    if cantidad_letras < 1 or cantidad_letras > 2:
        return "MAL"
    else:
        return "OK"

def verificar_condicion_final(crop_condicion_final):
    #Verificar campo condicion final ==============================================================
    caracteres_condicion_final, cantidad_letras = buscar_letras(crop_condicion_final, umbral=138, area_minima=2)
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
    return campo_condicion_final, calificacion

def guardar_resultados_por_hoja(hoja:int, registros_hoja:list[Record]):
    header = ["Id",
              "Legajo",
              "Nombre_apellido",
              "Parcial_1",
              "Parcial_2",
              "Parcial_3",
              "Condicion_final"
              ]
    data = []
    for rec in registros_hoja:
        data.append([rec.id,
                     rec.record_check.campo_legajo,
                     rec.record_check.campo_nombre,
                     rec.record_check.campo_parcial_1,
                     rec.record_check.campo_parcial_2,
                     rec.record_check.campo_parcial_3,
                     rec.record_check.campo_condicion_final])
    with open(str(OUTPUTS_DIR / f"validacion_grade_sheet_{hoja}.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def get_max_shape( actual, nuevo):
    return (max(nuevo[0], actual[0]),max(nuevo[1],actual[1]))

def guardar_desaprobados_por_hoja(hoja:int, registros_hoja:list[Record]):
    rec_out:list[Record]
    rec_out=[]

    max_nombre = (0,0)
    max_condic = (0,0)
    for rec in registros_hoja:
        if rec.record_check.registro_valido and rec.record_check.calificacion in ["R", "L"]:
            max_nombre = get_max_shape( max_nombre, rec.record_crops.crop_nombre.shape )
            max_condic = get_max_shape( max_condic, rec.record_crops.crop_condicion_final.shape )
            rec_out.append(rec)
    n = len(rec_out)

    margen = 20
    inter = 6
    alto_registro = max(max_nombre[0],max_condic[0]) + inter
    alto_salida = 2 * margen + alto_registro * n
    ancho_salida = max(500, 3 * margen + max_nombre[1] + max_condic[1])
 
    img_salida = np.full((alto_salida, ancho_salida, 3), 255, dtype=np.uint8)
    row = 0
    for rec in rec_out:
        y = margen + row * alto_registro
        x1 = margen
        x2 = margen * 2 + max_nombre[1]
        put_crop(y, x1, img_salida, rec.record_crops.crop_nombre)
        put_crop(y, x2, img_salida, rec.record_crops.crop_condicion_final)
        row = row + 1

    ruta_salida = OUTPUTS_DIR / f"validacion_grade_sheet_{hoja}.na.png"
    if not cv2.imwrite(str(ruta_salida), img_salida):
        raise OSError(f"No se pudo guardar la imagen: {ruta_salida}")

def guardar_validos(hoja:int, registros_hoja:list[Record]):
    rec_val:list[Record]=[]
    rec_inv:list[Record]=[]

    max_legajo = (0,0)
    max_nombre = (0,0)
    max_parc_1 = (0,0)
    max_parc_2 = (0,0)
    max_parc_3 = (0,0)
    max_condic = (0,0)
            
    for rec in registros_hoja:
        max_legajo = get_max_shape( max_legajo, rec.record_crops.crop_legajo.shape )
        max_nombre = get_max_shape( max_nombre, rec.record_crops.crop_nombre.shape )
        max_parc_1 = get_max_shape( max_parc_1, rec.record_crops.crop_parcial_1.shape )
        max_parc_2 = get_max_shape( max_parc_2, rec.record_crops.crop_parcial_2.shape )
        max_parc_3 = get_max_shape( max_parc_3, rec.record_crops.crop_parcial_3.shape )
        max_condic = get_max_shape( max_condic, rec.record_crops.crop_condicion_final.shape )
        if rec.record_check.registro_valido:
            rec_val.append(rec)
        else:
            rec_inv.append(rec)
    guardar_validos_out(hoja, rec_val, "valid", max_legajo, max_nombre, max_parc_1, max_parc_2, max_parc_3, max_condic)
    guardar_validos_out(hoja, rec_inv, "invalid", max_legajo, max_nombre, max_parc_1, max_parc_2, max_parc_3, max_condic)

def guardar_validos_out(hoja:int, rec_out:list[Record], suffix, 
                        max_legajo, max_nombre, max_parc_1, max_parc_2, max_parc_3, max_condic):
    
    n = len(rec_out)
    margen = 20
    inter = 6
    alto_registro = max(max_legajo[0], max_nombre[0], max_parc_1[0], max_parc_2[0], max_parc_3[0], max_condic[0]) + inter
    alto_salida = 2 * margen + alto_registro * n
    ancho_salida = max(500, 7 * margen + max_legajo[1]+max_nombre[1]+max_parc_1[1]+max_parc_2[1]+max_parc_3[1]+max_condic[1])
 
    img_salida = np.full((alto_salida, ancho_salida, 3), 255, dtype=np.uint8)
    row = 0
    for rec in rec_out:
        y = margen + row * alto_registro
        x1 = margen
        x2 = x1 + max_legajo[1] + margen
        x3 = x2 + max_nombre[1] + margen
        x4 = x3 + max_parc_1[1] + margen
        x5 = x4 + max_parc_2[1] + margen
        x6 = x5 + max_parc_3[1] + margen
        put_crop(y, x1, img_salida, rec.record_crops.crop_legajo)
        put_crop(y, x2, img_salida, rec.record_crops.crop_nombre)
        put_crop(y, x3, img_salida, rec.record_crops.crop_parcial_1)
        put_crop(y, x4, img_salida, rec.record_crops.crop_parcial_2)
        put_crop(y, x5, img_salida, rec.record_crops.crop_parcial_3)
        put_crop(y, x6, img_salida, rec.record_crops.crop_condicion_final)
        row = row + 1

    ruta_salida = OUTPUTS_DIR / f"validacion_grade_sheet_{hoja}.{suffix}.png"
    if not cv2.imwrite(str(ruta_salida), img_salida):
        raise OSError(f"No se pudo guardar la imagen: {ruta_salida}")

def put_crop(y:int, x:int, img_salida, crop):
    crop_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    img_salida[y:y + crop_color.shape[0], x: x + crop_color.shape[1]] = crop_color
      

def main():
    #========================================================================================
    #Comienzo del código para procesar las hojas de calificaciones
    #========================================================================================

    
    hojas = ["1", "2", "3", "4"]
    registros = []
#    registro_valido = False

    #Itero sobre cada hoja de calificaciones
    for hoja in hojas:
        registros_hoja = []
        img = cv2.imread(
            SOURCE_DIR / f"grade_sheet_{hoja}.png",
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
        print(f"\nProcesando hoja {hoja}...")
        for i in range(1, len(linea_horizontal) - 1):

            linea_superior = linea_horizontal[i] + 2
            linea_inferior = linea_horizontal[i + 1] - 2

            crop_legajo = img[linea_superior:linea_inferior, linea_vertical[1]+2:linea_vertical[2]-2]
            crop_nombre = img[linea_superior:linea_inferior, linea_vertical[2]+2:linea_vertical[3]-2]
            crop_parcial_1 = img[linea_superior:linea_inferior, linea_vertical[3]+2:linea_vertical[4]-2]
            crop_parcial_2 = img[linea_superior:linea_inferior, linea_vertical[4]+2:linea_vertical[5]-2]
            crop_parcial_3 = img[linea_superior:linea_inferior, linea_vertical[5]+2:linea_vertical[6]-2]
            crop_condicion_final = img[linea_superior:linea_inferior, linea_vertical[6]+2:linea_vertical[7]-2]

            record_crops = Record_crops(crop_legajo,crop_nombre,crop_parcial_1,crop_parcial_2,crop_parcial_3,crop_condicion_final)
            #for debug
            #record_crops.save(hoja,i)

            campo_legajo = verificar_legajo(crop_legajo)
            campo_nombre = verificar_nombre(crop_nombre)
            campo_parcial_1 = verificar_parcial(crop_parcial_1)
            campo_parcial_2 = verificar_parcial(crop_parcial_2)
            campo_parcial_3 = verificar_parcial(crop_parcial_3)
            campo_condicion_final, calificacion = verificar_condicion_final(crop_condicion_final)

            record_check = Record_check(campo_legajo, campo_nombre, campo_parcial_1, campo_parcial_2, campo_parcial_3, campo_condicion_final, calificacion)
            record_check.report(i)

            registros_hoja.append( Record(i, record_crops, record_check))

            registros.append({
                "Hoja": hoja,
                "Id": i,
                "Legajo": campo_legajo,
                "Nombre_apellido": campo_nombre,
                "Parcial_1": campo_parcial_1,
                "Parcial_2": campo_parcial_2,
                "Parcial_3": campo_parcial_3,
                "Condicion_final": campo_condicion_final,
                "Registro_valido": record_check.registro_valido,
                "Condicion_final_letra": calificacion,
                "a": linea_superior,
                "b": linea_inferior,
                "c": linea_vertical[2]+2,
                "d": linea_vertical[3]-2,
            })
        
        guardar_resultados_por_hoja(hoja, registros_hoja)
        guardar_desaprobados_por_hoja(hoja, registros_hoja)
        guardar_validos(hoja, registros_hoja)


    df_imagen = pd.DataFrame(
            registros,
            columns=[
                "Hoja",            
                "Registro_valido",
                "Condicion_final_letra",
                "a",
                "b",
                "c",
                "d"
            ],
        )

    df_imagen = df_imagen[
        (df_imagen["Registro_valido"] == True) & 
        ((df_imagen["Condicion_final_letra"] == "R") | 
         (df_imagen["Condicion_final_letra"] == "L"))
    ]

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
            "Condicion_final"
        ],
    )

    
    df_resultados.to_csv(
        OUTPUTS_DIR / "validacion_resultados.csv",
        index=False,
    )
    

    print("\nResultados guardados en validacion_resultados.csv")
    print(df_imagen.head(15).to_string(index=False))
    generar_imagen(df_imagen)
    print(f"Imagen de no aprobados guardada en {OUTPUTS_DIR / 'no_aprobados.png'}")


if __name__ == "__main__":
    main()

