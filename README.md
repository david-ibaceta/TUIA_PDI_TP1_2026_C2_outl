UNR | TUIA | PDI 

TRABAJO PRÁCTICO N° 1 - Año 2026 - 2° Semestre
======================

# Preparación

Para ejecutar el código alojado en la carpeta src, será necesario crear un entorno virtual. Por ejemplo, desde la raiz del repositorio clonado:

    ~/TUIA_PDI_TP1_2026_C2_outl$ python3 -m venv myenv

Esto resultará en la creación de una carpeta "myenv" donde se alijaran los artefactos a emplear en este proyecto (librerías, artefactos, etc)

La actrivación del ambiente se logra ejecutando un script alojado en el ambiente:

    ~/TUIA_PDI_TP1_2026_C2_outl$ source myenv/bin/activate

y se distingue con un prefijo en el prompt:

    (myenv) ~/TUIA_PDI_TP1_2026_C2_outl$

Con el ambiente activado se puede proceder a instalar (en este ambiente virtual), las librerías necesarias:

    pip install numpy
    pip install matplotlib
    pip install opencv-python
    pip install pandas

Hecho esto se puede proceder a la ejecución de las soluciones de los problemas

# Problema 1 - Ecualización local de histograma

Con el ambiente activado ejecutar:

    (myenv) ~/TUIA_PDI_TP1_2026_C2_outl$ python TP1/src/TP_PDI_problema_1.py 

(Atención, desde el directorio TP1/src, el camino hacia el archvo a ejecutar es distinto)

Esta ejecución emplea imágenes alojadas el TP1/source; en este caso

    Imagen_con_detalles_escondidos.tif

Asimismo, genera imagenes que se van a alojar en TP1/outputs

    imagen_original.png
    ecualizacion_local_3x3.png
    ecualizacion_local_5x5.png 
    ...

El procesamiento de la imagen consiste en aplicar una ecualización local. 

# Problema 2 - Validación de planilla de calificaciones

Con el ambiente activado ejecutar:

    (myenv) ~/TUIA_PDI_TP1_2026_C2_outl$ python TP1/src/TP_PDI_problema_2.py 

(Atención, desde el directorio TP1/src, el camino hacia el archvo a ejecutar es distinto)

Esta ejecución emplea imágenes alojadas el TP1/source; en este caso

    grade_sheet_1.png
    grade_sheet_2.png
    grade_sheet_3.png
    grade_sheet_4.png

Asimismo, genera archivos de texto e imágenes imagenes que se van a alojar en TP1/outputs

    validacion_grade_sheet_1.csv
    validacion_grade_sheet_1.na.png
    validacion_grade_sheet_1.invalid.png
    validacion_grade_sheet_1.valid.png
    ...

El procesamiento de cada una de las imagenes consiste en detectar los casilleros de la lista de notas y verificar su contenido de acuerdo a los controles especificados.

Cada una de estas imágenes es una version distinta de una lista de notas y condiciones. 

Para cada una se genera un archivo de texto **validacion_grade_sheet_{n}.csv** que contiene los resultados de evaluación de cada campo considerado.

Se genera asimismo una lista de los alumnos que no aprobaron el curso, y su correspondiente condición final **validacion_grade_sheet_{n}.na.png**.

Para facilitar la evaluación de los resultados se agregan dos imagenes conteniendo los registros válidos **validacion_grade_sheet_{n}.valid.png** e inválidos **validacion_grade_sheet_{n}.invalid.png**.

También se incluye un archivo global con las evaluaciones **validacion_resultados.csv** y otra versión, también global de quienes no aprobaron **no_aprobados.png**.

Hay un proceso destinado a debugging que alamcena los recortes obetenidos en la carpeta **TP1/debug**