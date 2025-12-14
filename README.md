# Detección y Conteo de Dados

Este proyecto implementa un **pipeline completo de procesamiento de video** para analizar una tirada de dados sobre una mesa con fondo verde. A partir de un video de entrada:

1. Detecta automáticamente la región de interés (ROI) correspondiente al fondo verde.
2. Recorta el video a dicha región.
3. Identifica los frames donde los dados están en reposo.
4. Detecta cada dado mediante segmentación por color.
5. Cuenta los puntos visibles en la cara superior de cada dado.
6. Genera un video de salida con bounding boxes, etiquetas individuales y resultados.
7. Muestra por consola los frames donde los dados se encuentran en reposo, el valor de cada dado y la suma total.


## Tecnologías utilizadas

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib (solo para pruebas y depuración)



## Estructura del proyecto

```
📁 TUIA-PDI-TP3/
│
├── tirada_1.mp4
├── tirada_2.mp4
├── tirada_3.mp4
├── tirada_4.mp4
├── TP3.py                      # Script principal
├── README.md
├── Informe TP3.pdf
├── TUIA_PDI_TP3_2025_C2.pdf
```


## Uso

1. Tener las librerias necesarias OpenCV (cv2), numpy, matplotlib.
2. Tener los videos de entrada en la carpeta actual con el nombre de tipo 'tirada_NUMERO.mp4' ('tirada_1.mp4').
3. Ejecutar el script principal:

```bash
python TP3.py
```

El programa:
- Procesa cada video
- Muestra resultados por consola
- Genera un video de salida con anotaciones
- Guarda una imagen del primer frame detenido con resultados
---

## Resultados

En los frames donde los dados están detenidos se muestra:

- Bounding box alrededor de cada dado
- Identificador único (Dado 1, Dado 2, ...)
- Valor de la cara superior

---

## Autores

ESTEVA MATIAS
PRIETO TOBIAS

Trabajo práctico realizado como parte de la materia **Procesamiento de Imágenes de la Tecnicatura en Inteligencia Artificial**. Año 2025.


