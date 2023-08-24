# funcion para detectar 

import cv2

import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



def detectar(frame, net_cara, net_tema):
    # se tomen las dimensiones del frame y se construye un blob desde ahi
    h, w=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pasa el blob por la red y se obtiene la deteccion de caras
    net_cara.setInput(blob)
    detecciones=net_cara.forward()

    # listas de caras, localizaciones y predicciones
    caras=[]
    locs=[]
    preds=[]

    # bucle sobre las detecciones
    for i in range(0, detecciones.shape[2]):
        # probabilidad asociada a la deteccion, umbral de confianza
        confianza=detecciones[0, 0, i, 2]

        # detecciones mayor que un umbral de confianza
        if confianza>0.5:
            
            # coordenadas (x, y) del contorno de la caja del objeto
            caja=detecciones[0, 0, i, 3:7]*np.array([w, h, w, h])
            
            x_start, y_start, x_end, y_end=caja.astype('int')

            # asegurar que los limites de la caja estan en el frame
            x_start, y_start=(max(0, x_start), max(0, y_start))
            x_end, y_end=(min(w-1, x_end), min(h-1, y_end))

            # extraer ROI de cara, pasar a RGB, redimensionar a 224x224 y preprocesar
            cara=frame[y_start:y_end, x_start:x_end]
            cara=cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
            cara=cv2.resize(cara, (224, 224))
            cara=img_to_array(cara)
            cara=preprocess_input(cara)

            # añadir a las listas
            caras.append(cara)
            locs.append((x_start, y_start, x_end, y_end))

    # solo se hacen predicciones si hay una cara detectada
    if len(caras)>0:
        # todas las caras a la vez
        caras=np.array(caras, dtype='float32')
        preds=net_tema.predict(caras, batch_size=32)

        
    return locs, preds
