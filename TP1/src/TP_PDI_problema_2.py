import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 1. Encontrar la ruta absoluta del proyecto para poder acceder a las carpetas 'source' y 'outputs'
# __file__ obtiene la posición de main.py. .parent nos saca de 'scr/' y nos deja en la raíz 'TP_PDI'
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Definir las rutas de las carpetas 'source' y 'outputs' usando la ruta base
SOURCE_DIR = BASE_DIR / "source"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Asegurar que la carpeta 'outputs' exista en la pc de los colaboradores
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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
    - umbral_simetria: umbral para la detección letra A (simetria
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

def main():
  #========================================================================================
  #Comienzo del código para procesar las hojas de calificaciones
  #========================================================================================

  
  hojas = ["1", "2", "3", "4"]
  registros = []
  registro_valido = False

  #Itero sobre cada hoja de calificaciones
  for hoja in hojas:
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

          print(f"\nRegistro {i}:")
          #Verificar campo Legajo ==============================================================
          _, cantidad_legajo = buscar_letras(crop_legajo, umbral=138, area_minima=3)
          if cantidad_legajo != 8:
              campo_legajo = "MAL"
          else:
              campo_legajo = "OK"
          print(f"Legajo: {campo_legajo}")

          #Verificar campo Nombre y apellido ======================================================
          caracteres_nombre, cantidad_letras = buscar_letras(
              crop_nombre, umbral=138, area_minima=6
          )
          cantidad_palabras = contar_palabras(caracteres_nombre, umbral_espacio = 6)

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
              crop_parcial_2, umbral=138, area_minima=2
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
              crop_parcial_3, umbral=138, area_minima=2
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
              crop_condicion_final, umbral=138, area_minima=2
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

          # Registrar si todos los registros cumplen condición "OK"
          if campo_legajo == "OK" and campo_nombre == "OK" and campo_parcial_1 == "OK" and campo_parcial_2 == "OK" and campo_parcial_3 == "OK" and campo_condicion_final == "OK":
            registro_valido = True
          else:
            registro_valido = False

          registros.append({
              "Hoja": hoja,
              "Id": i,
              "Legajo": campo_legajo,
              "Nombre_apellido": campo_nombre,
              "Parcial_1": campo_parcial_1,
              "Parcial_2": campo_parcial_2,
              "Parcial_3": campo_parcial_3,
              "Condicion_final": campo_condicion_final,
              "Registro_valido": registro_valido
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
          "Registro_valido"
      ],
  )

  
  df_resultados.to_csv(
      OUTPUTS_DIR / "validacion_resultados.csv",
      index=False,
  )
 

  print("\nResultados guardados en validacion_resultados.csv")


if __name__ == "__main__":
    main()