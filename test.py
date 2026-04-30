import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import re

# --- CONFIGURACIÓN ---
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "tu_foto.jpg" 
CONF_THRESHOLD = 0.30 # Bajamos un poco para probar

def validar_formato(texto):
    return re.sub(r"[^A-Z0-9]", "", texto.upper()).strip()

def main():
    # 1. Cargar Interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, in_h, in_w, _ = input_details[0]['shape']

    # 2. Leer Imagen
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: No se encontró la imagen.")
        return
    h_orig, w_orig = img.shape[:2]

    # 3. Preparar entrada
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(rgb, (in_w, in_h))
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # 4. Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    
    # YOLOv8 export: si el output es [5, 2100], transponer
    if output.shape[0] == 5:
        output = output.T

    print(f"Total de candidatos analizados: {len(output)}")
    
    encontrados = 0
    for i, det in enumerate(output):
        conf = det[4]
        
        if conf > CONF_THRESHOLD:
            encontrados += 1
            cx, cy, w, h = det[:4]

            # --- ESCALADO MANUAL DIRECTO ---
            # Probamos asumiendo que las coordenadas vienen en escala 0 a input_w
            x1 = int((cx - w/2) * (w_orig / in_w))
            y1 = int((cy - h/2) * (h_orig / in_h))
            x2 = int((cx + w/2) * (w_orig / in_w))
            y2 = int((cy + h/2) * (h_orig / in_h))

            # Asegurar que estén dentro de la imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            print(f"[{encontrados}] Detectada placa con conf {conf:.2f} en coords: {x1, y1, x2, y2}")

            # Recorte para OCR
            roi = img[y1:y2, x1:x2]
            texto = ""
            if roi.size > 0:
                # Pre-proceso ultra simple para test
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                texto = pytesseract.image_to_string(roi_gray, config="--psm 7")
                texto = validar_formato(texto)

            # Dibujar siempre que supere el umbral
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{texto} ({conf:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if encontrados == 0:
        print("El modelo no encontró nada por encima del umbral.")
        # Tip de diagnóstico: imprime la confianza más alta
        print(f"Confianza máxima en todo el output: {np.max(output[:, 4]):.4f}")
    else:
        cv2.imwrite("test_debug.jpg", img)
        print(f"Hecho. Se dibujaron {encontrados} cuadros en 'test_debug.jpg'")

if __name__ == "__main__":
    main()
