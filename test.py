import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract

# --- CONFIGURACIÓN ---
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "tu_foto_de_placa.jpg"
CONF_THRESHOLD = 0.35

def limpiar_placa(roi):
    """Optimiza el recorte de la placa para mejorar el OCR"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Aumentar contraste y binarizar (Blanco y Negro puro)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def main():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, input_h, input_w, _ = input_details[0]['shape']

    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None: return
    h_orig, w_orig = original_img.shape[:2]

    # Pre-proceso para el modelo
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(rgb_img, (input_w, input_h))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    if output.shape[0] == 5: output = output.T

    for detection in output:
        conf = detection[4]
        if conf > CONF_THRESHOLD:
            cx, cy, w, h = detection[:4]
            
            # Cálculo de coordenadas
            if np.max(detection[:4]) <= 1.1:
                x1, y1 = int((cx-w/2)*w_orig), int((cy-h/2)*h_orig)
                x2, y2 = int((cx+w/2)*w_orig), int((cy+h/2)*h_orig)
            else:
                x1, y1 = int((cx-w/2)*(w_orig/input_w)), int((cy-h/2)*(h_orig/input_h))
                x2, y2 = int((cx+w/2)*(w_orig/input_w)), int((cy+h/2)*(h_orig/input_h))

            # Ajustar bordes
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)

            # --- PARTE NUEVA: OCR ---
            # 1. Recortar la placa de la imagen original
            placa_roi = original_img[y1:y2, x1:x2]
            
            if placa_roi.size > 0:
                # 2. Limpiar imagen para Tesseract
                placa_limpia = limpiar_placa(placa_roi)
                
                # 3. Ejecutar OCR (Configuración para texto corto y alfanumérico)
                config_tess = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                texto_placa = pytesseract.image_to_string(placa_limpia, config=config_tess)
                texto_placa = texto_placa.strip()

                print(f"Detección: {texto_placa} (Conf: {conf:.2f})")

                # Dibujar resultados
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(original_img, f"ID: {texto_placa}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite("resultado_ocr.jpg", original_img)
    print("Proceso completo. Revisa resultado_ocr.jpg")

if __name__ == "__main__":
    main()
