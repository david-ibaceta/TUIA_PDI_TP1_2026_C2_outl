"""
Trabajo Práctico N° 1 - Año 2026 - 2° Semestre
Procesamiento de Imágenes I - IA 4.4
Tecnicatura Universitaria en Inteligencia Artificial (TUIA) - FCEIA - UNR

Problema 2 - Validación de planilla de calificaciones

Este script implementa:
a. Entrada: Imagen de planilla de calificaciones.
   Salida por terminal: Estado de cada campo (OK o MAL) por cada registro.
b. Generación de UNA ÚNICA IMAGEN DE SALIDA que informa aquellos alumnos que no hayan
   aprobado (Condición Final: "L" o "R"), considerando solamente los registros que se
   hayan cargado correctamente. Dicha imagen cuenta con el "crop" del contenido del
   campo Nombre y Apellido junto con un indicador que diferencia a los alumnos que
   deben recuperar ("R") de los que están en condición de libre ("L").
c. Generación de archivo CSV para almacenar los resultados de cada validación
   (ID, Legajo, Nombre y Apellido, Parcial 1, Parcial 2, Parcial 3, Condición Final)
   utilizando únicamente OK o MAL.
d. Aplicación cíclica sobre el conjunto de cuatro imágenes (grade_sheet_1.png a 4).
"""

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# 1. DETECCIÓN DE ESTRUCTURA Y GRILLA DE LA TABLA
# =============================================================================

def detectar_lineas_grilla(img_gray):
    """
    Detecta las coordenadas de las líneas horizontales y verticales de la tabla
    utilizando binarización, morfología matemática y análisis de perfiles de proyección.

    Retorna:
        y_filas_alumnos: lista de 21 coordenadas Y que delimitan las 20 filas de alumnos.
        x_cols: lista de 8 coordenadas X que delimitan las 7 columnas.
    """
    img_th = (img_gray < 128).astype(np.uint8)
    alto, ancho = img_gray.shape

    tam_h = max(25, ancho // 30)
    tam_v = max(25, alto // 30)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (tam_h, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, tam_v))

    h_lines = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel_h)
    v_lines = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel_v)

    row_sum = np.sum(h_lines, axis=1)
    col_sum = np.sum(v_lines, axis=0)

    def obtener_coordenadas_lineas(perfil, factor_umbral=0.35):
        umbral = factor_umbral * np.max(perfil)
        indices = np.where(perfil > umbral)[0]
        if len(indices) == 0:
            return []
        grupos = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        return [(g[0] + g[-1]) // 2 for g in grupos]

    y_lineas = obtener_coordenadas_lineas(row_sum, factor_umbral=0.35)
    x_lineas = obtener_coordenadas_lineas(col_sum, factor_umbral=0.35)

    # Las 20 filas de alumnos están delimitadas por las últimas 21 líneas horizontales
    if len(y_lineas) >= 21:
        y_filas_alumnos = y_lineas[-21:]
    else:
        y_filas_alumnos = y_lineas

    return y_filas_alumnos, x_lineas


# =============================================================================
# 2. SEGMENTACIÓN Y EXTRACCIÓN DE CARACTERES Y PALABRAS POR CELDA
# =============================================================================

def segmentar_contenido_celda(celda_gray, umbral_bin=135, th_area=2):
    """
    Segmenta y analiza los caracteres y palabras presentes en una celda.

    Resuelve de forma robusta:
    - Eliminación de bordes de la grilla mediante margen interior.
    - Celdas vacías (fondo uniforme blanco).
    - Fusión de trazos verticalmente superpuestos (como acentos o 'Ñ').
    - Descomposición de caracteres que se tocan debido al renderizado/anti-aliasing
      a través del ancho de las componentes conectadas.
    - Detección de separación de palabras (gaps >= 4 píxeles).

    Retorna:
        num_caracteres: int
        num_palabras: int
    """
    alto, ancho = celda_gray.shape
    pad_y = max(2, int(alto * 0.12))
    pad_x = max(2, int(ancho * 0.05))

    celda_interior = celda_gray[pad_y:alto - pad_y, pad_x:ancho - pad_x]
    if celda_interior.size == 0:
        return 0, 0

    # Si la celda está prácticamente vacía (blanca)
    pixeles_oscuros = np.sum(celda_interior < umbral_bin)
    if pixeles_oscuros < 5:
        return 0, 0

    celda_bin = (celda_interior < umbral_bin).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        celda_bin, connectivity=8
    )

    componentes = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= th_area:
            componentes.append([x, x + w, w, area])

    if not componentes:
        return 0, 0

    # Ordenar componentes de izquierda a derecha
    componentes.sort(key=lambda c: c[0])

    # Fusión de fragmentos que se solapan horizontalmente (ej. glifos con partes disjuntas como 'Ñ')
    fusionados = []
    for c in componentes:
        if not fusionados:
            fusionados.append(c)
        else:
            # Si el componente actual solapa horizontalmente con el anterior
            if c[0] < fusionados[-1][1]:
                fusionados[-1][1] = max(fusionados[-1][1], c[1])
                fusionados[-1][2] = fusionados[-1][1] - fusionados[-1][0]
                fusionados[-1][3] += c[3]
            else:
                fusionados.append(c)

    # Estimación precisa de caracteres considerando posibles caracteres que se tocan
    # Ancho típico de un caracter en esta tipografía: 4 a 6 px ('M' es 7 px, '-' o '/' son 2-3 px).
    # Si dos caracteres se tocan, el ancho fusionado es >= 8 px.
    num_caracteres = 0
    for f in fusionados:
        w = f[2]
        if w <= 7:
            num_caracteres += 1
        elif w <= 12:
            num_caracteres += 2  # 2 caracteres adosados (ej. '-' con '4')
        elif w <= 17:
            num_caracteres += 3  # 3 caracteres adosados (ej. '-44')
        elif w <= 22:
            num_caracteres += 4
        else:
            num_caracteres += max(1, int(round(w / 5.2)))

    if num_caracteres == 0:
        return 0, 0

    # Detección de separación de palabras:
    # Espaciado normal entre letras continuas: 0 a 2 píxeles.
    # Espacio real entre palabras (spacebar): >= 4 píxeles.
    gaps = [
        fusionados[i + 1][0] - fusionados[i][1]
        for i in range(len(fusionados) - 1)
    ]
    num_espacios = sum(1 for g in gaps if g >= 4)
    num_palabras = 1 + num_espacios

    return num_caracteres, num_palabras


# =============================================================================
# 3. REGLAS DE VALIDACIÓN DE CAMPOS (PÁGINA 2 DEL PDF)
# =============================================================================

def validar_campo_legajo(n_chars, n_words):
    """
    i. Legajo: Debe contener sólo 8 caracteres en total, formando una única palabra.
    """
    return (n_chars == 8) and (n_words == 1)


def validar_campo_nombre_apellido(n_chars, n_words):
    """
    ii. Nombre y apellido: Debe contener un mínimo de dos palabras y no más de 12 caracteres en total.
    """
    return (n_words >= 2) and (n_chars <= 12)


def validar_campo_parcial(n_chars, n_words):
    """
    iii. Notas (Parcial 1, 2 y 3): Cada campo debe contener 1 o 2 caracteres consecutivos
         (no deben existir espacios entre ellos).
    """
    return (n_chars in (1, 2)) and (n_words == 1)


def validar_campo_condicion_final(n_chars, n_words):
    """
    iv. Condición Final: Debe contener único caracter.
    """
    return (n_chars == 1) and (n_words == 1)


# =============================================================================
# 4. CLASIFICACIÓN DE CONDICIÓN FINAL ('R', 'L', 'A')
# =============================================================================

def clasificar_condicion_final(celda_gray):
    """
    Determina si el caracter de Condición Final es 'A', 'R' o 'L'.
    Utiliza propiedades morfológicas y topológicas:
    - 'L': no posee huecos interiores y su cuadrante superior derecho está vacío.
    - 'R': posee un bucle superior cerrado, trazo vertical a la izquierda y pata derecha.
    - 'A': posee bucle cerrado superior con lados diagonales y apertura en la base central.
    """
    alto, ancho = celda_gray.shape
    pad_y = max(2, int(alto * 0.10))
    pad_x = max(2, int(ancho * 0.04))
    celda_interior = celda_gray[pad_y:alto - pad_y, pad_x:ancho - pad_x]

    bin_img = (celda_interior < 140).astype(np.uint8) * 255
    contornos, jerarquia = cv2.findContours(
        bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return None

    contorno_max = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_max)
    glifo = bin_img[y:y + h, x:x + w]

    # Contar agujeros internos
    num_huecos = 0
    if jerarquia is not None:
        for j in jerarquia[0]:
            if j[3] != -1:
                num_huecos += 1

    # 'L' no tiene bucle interior y su cuadrante superior derecho está prácticamente vacío
    cuadrante_sup_der = glifo[:max(1, h // 2), max(1, w // 2):]
    densidad_sup_der = np.sum(cuadrante_sup_der > 0) / (cuadrante_sup_der.size + 1e-5)

    if num_huecos == 0 or densidad_sup_der < 0.08:
        return 'L'

    # Para distinguir 'R' de 'A':
    # La 'R' tiene una columna vertical continua a la izquierda (densidad alta en x < w/3)
    # y la 'A' es triangular / diagonal con base central abierta.
    col_izq = glifo[:, :max(2, w // 3)]
    densidad_izq = np.sum(col_izq > 0) / (col_izq.size + 1e-5)

    base_central = glifo[int(h * 0.65):, int(w * 0.25):int(w * 0.75)]
    densidad_base_central = np.sum(base_central > 0) / (base_central.size + 1e-5)

    if densidad_izq > 0.42 and densidad_base_central < 0.25:
        return 'R'
    elif densidad_izq > 0.45:
        return 'R'
    else:
        return 'A'


# =============================================================================
# 5. GENERACIÓN DE LA ÚNICA IMAGEN DE SALIDA (PUNTO B)
# =============================================================================

def generar_imagen_no_aprobados(no_aprobados, ruta_salida, titulo_superior="ALUMNOS NO APROBADOS"):
    """
    Genera UNA ÚNICA IMAGEN DE SALIDA que informa aquellos alumnos que no hayan
    aprobado (Condición Final: 'L' o 'R'), considerando solamente los registros
    que se hayan cargado correctamente.

    Contiene:
    - El "crop" del contenido del campo Nombre y Apellido.
    - Un indicador visual claro con código de colores que diferencia:
        * Alumnos que deben recuperar (Condición Final: 'R') en color Naranja/Ámbar.
        * Alumnos en condición de libre (Condición Final: 'L') en color Rojo Carmesí.
    """
    if not no_aprobados:
        img_info = np.full((120, 700, 3), 255, dtype=np.uint8)
        cv2.putText(
            img_info,
            "No se encontraron alumnos no aprobados en registros validos.",
            (30, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 128, 0),
            2
        )
        cv2.imwrite(str(ruta_salida), img_info)
        return

    alto_fila = 48
    ancho_crop = 230
    ancho_total = 750
    alto_encabezado = 80
    alto_total = alto_encabezado + len(no_aprobados) * alto_fila + 25

    lienzo = np.full((alto_total, ancho_total, 3), 255, dtype=np.uint8)

    # Encabezado principal
    cv2.rectangle(lienzo, (0, 0), (ancho_total, alto_encabezado), (35, 45, 55), -1)
    cv2.putText(
        lienzo,
        titulo_superior.upper(),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    cv2.putText(
        lienzo,
        "Registros correctamente validados con Condicion Final 'R' (Recupera) o 'L' (Libre)",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 205, 215),
        1
    )

    y_actual = alto_encabezado + 12

    for i, item in enumerate(no_aprobados):
        crop_nombre = item["crop_nombre"]
        condicion = item["condicion"]
        nro_registro = item["registro"]
        planilla_origen = item.get("planilla", "")

        # Fondo alternado
        if i % 2 == 0:
            cv2.rectangle(
                lienzo,
                (8, y_actual - 4),
                (ancho_total - 8, y_actual + alto_fila - 6),
                (245, 247, 250),
                -1
            )

        # Identificador de registro y planilla
        etiqueta_id = f"{planilla_origen} Reg. {nro_registro:02d}:" if planilla_origen else f"Reg. {nro_registro:02d}:"
        cv2.putText(
            lienzo,
            etiqueta_id,
            (15, y_actual + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 60, 60),
            1
        )

        # Pegar crop de Nombre y Apellido
        offset_crop_x = 180 if planilla_origen else 115
        h_crop, w_crop = crop_nombre.shape[:2]
        if h_crop > 0 and w_crop > 0:
            escala = min(28 / h_crop, ancho_crop / w_crop)
            nuevo_w = max(1, int(w_crop * escala))
            nuevo_h = max(1, int(h_crop * escala))
            crop_resized = cv2.resize(
                crop_nombre, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA
            )

            if len(crop_resized.shape) == 2:
                crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
            else:
                crop_bgr = crop_resized

            offset_y = y_actual + (alto_fila - 10 - nuevo_h) // 2
            lienzo[offset_y:offset_y + nuevo_h, offset_crop_x:offset_crop_x + nuevo_w] = crop_bgr
            cv2.rectangle(
                lienzo,
                (offset_crop_x - 2, offset_y - 2),
                (offset_crop_x + nuevo_w + 1, offset_y + nuevo_h + 1),
                (200, 200, 200),
                1
            )

        # Indicador diferenciador de Condición
        # 'R': Naranja / Ámbar -> RECUPERA
        # 'L': Rojo Carmesí -> LIBRE
        offset_badge_x = 450
        offset_badge_y = y_actual + 7
        ancho_badge = 270
        alto_badge = 28

        if condicion == 'R':
            color_badge = (34, 126, 230)   # BGR: Naranja (#E67E22)
            texto_badge = "RECUPERA (Condicion: R)"
        else:
            color_badge = (43, 57, 192)    # BGR: Rojo (#C0392B)
            texto_badge = "LIBRE (Condicion: L)"

        cv2.rectangle(
            lienzo,
            (offset_badge_x, offset_badge_y),
            (offset_badge_x + ancho_badge, offset_badge_y + alto_badge),
            color_badge,
            -1
        )
        cv2.putText(
            lienzo,
            texto_badge,
            (offset_badge_x + 18, offset_badge_y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )

        y_actual += alto_fila

    cv2.imwrite(str(ruta_salida), lienzo)


# =============================================================================
# 6. FUNCIÓN DE VALIDACIÓN DE UNA PLANILLA (PUNTOS A Y C)
# =============================================================================

def validar_planilla(ruta_imagen, ruta_csv_salida=None, ruta_img_salida=None, imprimir_terminal=True):
    """
    Procesa y valida una imagen de planilla de calificaciones.

    Parámetros:
        ruta_imagen: str o Path a la imagen del formulario.
        ruta_csv_salida: str o Path donde guardar el archivo CSV de validaciones.
        ruta_img_salida: str o Path donde guardar la imagen con alumnos no aprobados de esta planilla.
        imprimir_terminal: bool, si muestra la validación campo a campo en terminal.

    Retorna:
        resultados: lista de diccionarios con las validaciones de cada registro.
        no_aprobados: lista de alumnos no aprobados con registros válidos.
    """
    ruta_imagen = Path(ruta_imagen)
    nombre_archivo = ruta_imagen.name

    img_bgr = cv2.imread(str(ruta_imagen))
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    y_filas, x_cols = detectar_lineas_grilla(img_gray)
    num_filas = len(y_filas) - 1
    num_cols = len(x_cols) - 1

    if num_filas < 20 or num_cols < 7:
        raise ValueError(
            f"No se detectó la cantidad esperada de celdas en {nombre_archivo}. "
            f"Filas: {num_filas}, Columnas: {num_cols}"
        )

    resultados = []
    no_aprobados = []

    if imprimir_terminal:
        print(f"\n{'='*60}")
        print(f"RESULTADOS DE VALIDACIÓN: {nombre_archivo}")
        print(f"{'='*60}")

    for r in range(20):
        id_registro = r + 1
        y_top = y_filas[r]
        y_bottom = y_filas[r + 1]

        # Col 1: Legajo
        celda_leg = img_gray[y_top:y_bottom, x_cols[1]:x_cols[2]]
        n_chars_leg, n_words_leg = segmentar_contenido_celda(celda_leg)
        ok_leg = validar_campo_legajo(n_chars_leg, n_words_leg)

        # Col 2: Nombre y Apellido
        celda_nom = img_gray[y_top:y_bottom, x_cols[2]:x_cols[3]]
        n_chars_nom, n_words_nom = segmentar_contenido_celda(celda_nom)
        ok_nom = validar_campo_nombre_apellido(n_chars_nom, n_words_nom)

        # Col 3: Parcial 1
        celda_p1 = img_gray[y_top:y_bottom, x_cols[3]:x_cols[4]]
        n_chars_p1, n_words_p1 = segmentar_contenido_celda(celda_p1)
        ok_p1 = validar_campo_parcial(n_chars_p1, n_words_p1)

        # Col 4: Parcial 2
        celda_p2 = img_gray[y_top:y_bottom, x_cols[4]:x_cols[5]]
        n_chars_p2, n_words_p2 = segmentar_contenido_celda(celda_p2)
        ok_p2 = validar_campo_parcial(n_chars_p2, n_words_p2)

        # Col 5: Parcial 3
        celda_p3 = img_gray[y_top:y_bottom, x_cols[5]:x_cols[6]]
        n_chars_p3, n_words_p3 = segmentar_contenido_celda(celda_p3)
        ok_p3 = validar_campo_parcial(n_chars_p3, n_words_p3)

        # Col 6: Condición Final
        celda_cond = img_gray[y_top:y_bottom, x_cols[6]:x_cols[7]]
        n_chars_cond, n_words_cond = segmentar_contenido_celda(celda_cond)
        ok_cond = validar_campo_condicion_final(n_chars_cond, n_words_cond)

        estado_registro = {
            "ID": id_registro,
            "Legajo": "OK" if ok_leg else "MAL",
            "Nombre y Apellido": "OK" if ok_nom else "MAL",
            "Parcial 1": "OK" if ok_p1 else "MAL",
            "Parcial 2": "OK" if ok_p2 else "MAL",
            "Parcial 3": "OK" if ok_p3 else "MAL",
            "Condición Final": "OK" if ok_cond else "MAL",
        }
        resultados.append(estado_registro)

        # Salida por pantalla (Punto a)
        if imprimir_terminal:
            print(f"> Registro {id_registro}:")
            print(f"> Legajo: {estado_registro['Legajo']}")
            print(f"> Nombre y apellido: {estado_registro['Nombre y Apellido']}")
            print(f"> Parcial 1: {estado_registro['Parcial 1']}")
            print(f"> Parcial 2: {estado_registro['Parcial 2']}")
            print(f"> Parcial 3: {estado_registro['Parcial 3']}")
            print(f"> Condición Final: {estado_registro['Condición Final']}")
            print(">")

        # Punto b: Solo registros que se hayan cargado correctamente (todos OK)
        registro_es_valido = all(
            estado_registro[campo] == "OK"
            for campo in [
                "Legajo",
                "Nombre y Apellido",
                "Parcial 1",
                "Parcial 2",
                "Parcial 3",
                "Condición Final",
            ]
        )

        if registro_es_valido:
            condicion_letra = clasificar_condicion_final(celda_cond)
            if condicion_letra in ("R", "L"):
                pad_y = max(2, int((y_bottom - y_top) * 0.10))
                pad_x = max(2, int((x_cols[3] - x_cols[2]) * 0.04))
                crop_nombre = img_bgr[
                    y_top + pad_y:y_bottom - pad_y,
                    x_cols[2] + pad_x:x_cols[3] - pad_x
                ]
                no_aprobados.append({
                    "registro": id_registro,
                    "crop_nombre": crop_nombre,
                    "condicion": condicion_letra,
                    "planilla": nombre_archivo
                })

    # Generación de archivo CSV (Punto c)
    if ruta_csv_salida:
        ruta_csv_salida = Path(ruta_csv_salida)
        ruta_csv_salida.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_csv_salida, mode="w", newline="", encoding="utf-8") as f:
            columnas = [
                "ID",
                "Legajo",
                "Nombre y Apellido",
                "Parcial 1",
                "Parcial 2",
                "Parcial 3",
                "Condición Final"
            ]
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            for fila in resultados:
                writer.writerow(fila)
        if imprimir_terminal:
            print(f"[+] Archivo CSV generado en: {ruta_csv_salida}")

    # Generación de imagen individual si fue solicitada
    if ruta_img_salida:
        ruta_img_salida = Path(ruta_img_salida)
        ruta_img_salida.parent.mkdir(parents=True, exist_ok=True)
        generar_imagen_no_aprobados(
            no_aprobados,
            ruta_img_salida,
            titulo_superior=f"ALUMNOS NO APROBADOS - {nombre_archivo}"
        )

    return resultados, no_aprobados


# =============================================================================
# 7. EJECUCIÓN CÍCLICA SOBRE EL CONJUNTO DE PLANILLAS (PUNTO D)
# =============================================================================

def ejecutar_procesamiento_ciclico():
    """
    Aplica el algoritmo desarrollado, de forma cíclica, sobre el conjunto
    de cuatro imágenes de planillas (grade_sheet_1.png a grade_sheet_4.png)
    e informa los resultados.

    Genera:
    - Archivos CSV individuales para cada planilla (Punto c).
    - UNA ÚNICA IMAGEN DE SALIDA consolidada con todos los alumnos no aprobados
      ('L' o 'R') de los registros válidos de todas las planillas (Punto b).
    """
    script_dir = Path(__file__).resolve().parent
    posibles_directorios = [
        script_dir.parent / "source",
        script_dir / "source",
        script_dir.parent,
        script_dir,
        Path("TP1/source"),
        Path("TP1"),
        Path("source"),
    ]

    source_dir = None
    for d in posibles_directorios:
        if (d / "grade_sheet_1.png").exists():
            source_dir = d
            break

    if source_dir is None:
        raise FileNotFoundError("No se encontró el directorio con las imágenes grade_sheet_<id>.png")

    output_dir = script_dir.parent / "outputs" if script_dir.name == "src" else script_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # También asegurar outputs en la raíz del proyecto para conveniencia
    output_dir_raiz = Path("outputs")
    output_dir_raiz.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRABAJO PRÁCTICO N° 1 - PROBLEMA 2: VALIDACIÓN DE PLANILLAS")
    print(f"Directorio de origen: {source_dir}")
    print(f"Directorio de salida: {output_dir}")
    print("=" * 70)

    imagenes_planillas = [
        source_dir / f"grade_sheet_{i}.png" for i in range(1, 5)
    ]

    resumen_global = []
    todos_los_no_aprobados = []

    for ruta_img in imagenes_planillas:
        if not ruta_img.exists():
            print(f"[!] Archivo no encontrado: {ruta_img}")
            continue

        id_sheet = ruta_img.stem
        csv_salida = output_dir / f"validacion_{id_sheet}.csv"
        csv_salida_raiz = output_dir_raiz / f"validacion_{id_sheet}.csv"

        resultados, no_aprobados = validar_planilla(
            ruta_img,
            ruta_csv_salida=csv_salida,
            imprimir_terminal=True
        )

        # Copiar CSV a la carpeta outputs raíz también
        if csv_salida.exists() and csv_salida != csv_salida_raiz:
            import shutil
            shutil.copy(csv_salida, csv_salida_raiz)

        # Acumular para la ÚNICA imagen de salida
        todos_los_no_aprobados.extend(no_aprobados)

        total_correctos = sum(
            1 for r in resultados
            if all(r[c] == "OK" for c in [
                "Legajo", "Nombre y Apellido", "Parcial 1",
                "Parcial 2", "Parcial 3", "Condición Final"
            ])
        )

        resumen_global.append({
            "planilla": ruta_img.name,
            "registros_validos": total_correctos,
            "registros_invalidos": len(resultados) - total_correctos,
            "no_aprobados_detectados": len(no_aprobados)
        })

    # Generar la ÚNICA imagen de salida que informa todos los no aprobados (Punto b)
    ruta_unica_imagen = output_dir / "alumnos_no_aprobados.png"
    ruta_unica_imagen_raiz = output_dir_raiz / "alumnos_no_aprobados.png"

    generar_imagen_no_aprobados(
        todos_los_no_aprobados,
        ruta_unica_imagen,
        titulo_superior="ALUMNOS NO APROBADOS (CONDICION 'R' O 'L')"
    )
    if ruta_unica_imagen != ruta_unica_imagen_raiz:
        import shutil
        shutil.copy(ruta_unica_imagen, ruta_unica_imagen_raiz)

    print("\n" + "=" * 70)
    print(f"[+] ÚNICA IMAGEN DE SALIDA generada exitosamente en:")
    print(f"    -> {ruta_unica_imagen}")
    print(f"    (Total de alumnos no aprobados diferenciados: {len(todos_los_no_aprobados)})")
    print("=" * 70)

    # Informe resumido al finalizar
    print("\n" + "=" * 70)
    print("RESUMEN GLOBAL DEL PROCESAMIENTO CÍCLICO")
    print("=" * 70)
    for res in resumen_global:
        print(
            f"- {res['planilla']:20s} | "
            f"Válidos: {res['registros_validos']:2d}/20 | "
            f"Inválidos: {res['registros_invalidos']:2d}/20 | "
            f"No Aprobados (R/L): {res['no_aprobados_detectados']:2d}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    ejecutar_procesamiento_ciclico()
