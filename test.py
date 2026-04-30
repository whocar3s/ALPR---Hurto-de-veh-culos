import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import re

# --- CONFIGURACIÓN ---
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "tu_foto.jpg" 
CONF_THRESHOLD = 0.25 # Umbral bajo para forzar detección

def main():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, in_h, in_w, _ = input_details[0]['shape']

    img = cv2.imread(IMAGE_PATH)
    if img is None: return
    h_orig, w_orig = img.shape[:2]

    # Pre-proceso
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(rgb, (in_w, in_h))
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

    # --- DIAGNÓSTICO DE ESTRUCTURA ---
    # YOLOv8 TFLite suele ser [Clases+4, 2100]. Transponemos si la primera dim es pequeña.
    if output.shape[0] < output.shape[1]:
        output = output.T
    
    print(f"Estructura del output: {output.shape}") # Debería ser algo como (2100, 5)

    # Encontrar la columna de confianza
    # En modelos de una sola clase (placa), la confianza suele estar en el índice 4
    # Pero vamos a buscar en CUALQUIER columna de la 4 en adelante.
    
    boxes_encontrados = []

    for row in output:
        # Buscamos el valor máximo desde la columna 4 en adelante (probabilidad de clase)
        probabilidades = row[4:]
        conf = np.max(probabilidades)
        
        if conf > CONF_THRESHOLD:
            # Si entramos aquí, el modelo REALMENTE vio algo
            cx, cy, w, h = row[:4]
            
            # Intentar dos tipos de escalado:
            # 1. Escalado si vienen normalizados (0-1)
            # 2. Escalado si vienen en tamaño de red (0-320)
            if np.max(row[:4]) <= 1.01:
                x1 = int((cx - w/2) * w_orig)
                y1 = int((cy - h/2) * h_orig)
                x2 = int((cx + w/2) * w_orig)
                y2 = int((cy + h/2) * h_orig)
            else:
                x1 = int((cx - w/2) * (w_orig / in_w))
                y1 = int((cy - h/2) * (h_orig / in_h))
                x2 = int((cx + w/2) * (w_orig / in_w))
                y2 = int((cy + h/2) * (h_orig / in_h))

            boxes_encontrados.append(((x1, y1, x2, y2), conf))

    # --- DIBUJO Y OCR ---
    if not boxes_encontrados:
        print("Sigo sin detectar nada. Confianza máxima encontrada:", np.max(output[:, 4:]))
    else:
        print(f"¡Se encontraron {len(boxes_encontrados)} posibles cuadros!")
        for (x1, y1, x2, y2), conf in boxes_encontrados:
            # Evitar cuadros basura (muy pequeños)
            if (x2 - x1) < 10 or (y2 - y1) < 10: continue
            
            # Asegurar límites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            # OCR Simple
            roi = img[y1:y2, x1:x2]
            texto = ""
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                texto = pytesseract.image_to_string(gray_roi, config="--psm 7").strip()

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{texto} {conf:.2f}", (x1, y1-5), 1, 1.5, (0, 255, 0), 2)

        cv2.imwrite("debug_final.jpg", img)
        print("Imagen guardada como debug_final.jpg")

if __name__ == "__main__":
    main()
