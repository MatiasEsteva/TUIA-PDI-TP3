import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


def detectar_roi(video):
    # Leer video 
    cap = cv2.VideoCapture(video)  # Abre el archivo de video especificado para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.

    frame_number = 0
    while (cap.isOpened()): # Verifica si el video se abrió correctamente.

        ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

        if ret == True:  

            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

            # Detectar fondo verde unicamente en el primer frame para obtener las coordenadas del ROI
            if frame_number == 0:
                frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(frame_hsv)

                # Detectar solo el verde
                ix_h = np.logical_and(h > 35, h < 100)
                ix_s = np.logical_and(s > 256 * 0.3, s < 256)
                ix = np.logical_and(ix_h, ix_s)

                # Mascara binaria
                mask = (ix.astype(np.uint8)) * 255

                # Encontrar contorno mas grande
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(contours, key=cv2.contourArea)
                x, y, w, h_roi = cv2.boundingRect(c)


            frame_crop = frame[y:y+h_roi, x:x+w]

            #cv2.imshow('Frame', frame_crop)   (Descomentar para ver todo el video recortado)

            frame_number += 1
            if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
                break
        else:  
            break  

    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.
    return (x, y, w, h_roi)


def encontrar_frames_quietos(video, roi):
    # Leer video 
    cap = cv2.VideoCapture(video)  # Abre el archivo de video especificado para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.

    x, y, w, h = roi
    frame_number = 0
    frame_prev = None
    quietos_count = 0
    UMBRAL_FRAMES = 12
    UMBRAL_MOVIMIENTO = 500
    frames_quietos = []

    while (cap.isOpened()): # Verifica si el video se abrió correctamente.

        ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

        if ret == True:  

            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

            frame_crop = frame[y:y+h, x:x+w] # Recortar el frame a las dimensiones obtenidas con la funcion anterior

            gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)

            if frame_prev is not None:
                diff = np.abs(blur.astype(np.int16) - frame_prev.astype(np.int16))

                motion_pixels = np.sum(diff > 25)

                if motion_pixels < UMBRAL_MOVIMIENTO:
                    quietos_count += 1
                else:
                    quietos_count = 0

                if quietos_count >= UMBRAL_FRAMES:
                    print(f"Dados detenidos en el frame: {frame_number}")
                    frames_quietos.append(frame_number)

            frame_prev = blur.copy()


            #cv2.imshow('Frame', frame_crop) # Muestra el frame redimensionado.

            frame_number += 1
            if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
                break
        else:  
            break  

    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.
    return frames_quietos


def procesar_frame_quieto(frame_crop):
    resultados = []
    img_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)

    # Segmentación rojo para obtener contornos de dados
    img_hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    ix_h1 = np.logical_and(h > 180 * 0.9, h < 180)
    ix_h2 = h < 180 * 0.04
    ix_s = np.logical_and(s > 256 * 0.3, s < 256)

    mask_red = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)

    # Mascara binaria
    mask_red = mask_red.astype(np.uint8) * 255

    # Rellenar huecos con clausura
    kernel = np.ones((7, 7), np.uint8)
    mask_solida = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    # Contornos externos
    contornos, _ = cv2.findContours(mask_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        return resultados

    # Buscar puntos blancos en el crop sobre la imagen en escala de grises de cada dado
    for c in contornos:

        if cv2.contourArea(c) < 50:
            continue

        x, y, w, h = cv2.boundingRect(c)
        roi_gray = img_gray[y:y+h, x:x+w] # Crear crop de cada dado en escala de grises

        # Umbralado
        _, mask_puntos = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)

        # Componentes conectadas para encontrar la cantidad de puntos
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_puntos, connectivity=8)

        numero_puntos = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 2:
                numero_puntos += 1

        if numero_puntos == 0:
            continue

        resultados.append(((x, y, w, h), numero_puntos))

    return resultados


def generar_video_salida(video, roi, frames_quietos):
    # Leer video 
    cap = cv2.VideoCapture(video)  # Abre el archivo de video especificado para su lectura.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
    fps = cap.get(cv2.CAP_PROP_FPS)

    x, y, w, h = roi

    # Video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    nombre_salida = 'resultado_'+video
    out = cv2.VideoWriter(nombre_salida, fourcc, fps, (w, h))
    
    frame_number = 0
    ya_impreso = False
    while (cap.isOpened()): # Verifica si el video se abrió correctamente.

        ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

        if ret == True:  

            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

            frame_crop = frame[y:y+h, x:x+w]

            if frame_number in frames_quietos:

                resultados = procesar_frame_quieto(frame_crop)

                for idx, ((x_d, y_d, w_d, h_d), numero) in enumerate(resultados, start=1):

                    etiqueta = f"Dado {idx}: {numero}"

                    cv2.rectangle(frame_crop,(x_d, y_d),(x_d + w_d, y_d + h_d),(0, 255, 0), 1)

                    cv2.putText(frame_crop, etiqueta, (x_d, y_d - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


                if not ya_impreso and len(resultados) > 0:
                    suma = sum(numero for (_, numero) in resultados)

                    for idx, (_, numero) in enumerate(resultados, start=1):
                        print(f"Dado {idx}: {numero}")

                    print(f"Suma total: {suma}")

                    nombre_img = f"imagen_resultado_{video[:8]}.png"
                    cv2.imwrite(nombre_img, frame_crop)
                    print(f"Imagen guardada: {nombre_img}")

                    ya_impreso = True

            cv2.imshow('Resultado', frame_crop) # Muestra el frame redimensionado.

            out.write(frame_crop)
            frame_number += 1

            if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
                break
        else:  
            break  

    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.



nombres_videos = []
for i in range(1,5):
    nombres_videos.append(f'tirada_{i}.mp4')



for video in nombres_videos:
    print(f'Procesando {video}...')
    roi = detectar_roi(video)
    frames_quietos = encontrar_frames_quietos(video, roi)
    generar_video_salida(video, roi, frames_quietos)
    print('\n')




