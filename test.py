import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract

# --- CONFIGURACIÓN ---
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "tu_foto_de_placa.jpg"  # Asegúrate de que el nombre coincida
CONF_THRESHOLD = 0.35

def mejorar_y_leer_placa(roi):
    """
    Aplica filtros avanzados para que Tesseract reconozca 
    caracteres en placas con fondo amarillo o ruido.
    """
    try:
        # 1. Escalado (Zoom): Tesseract necesita letras de buen tamaño
        roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Filtro Bilateral: Reduce ruido manteniendo bordes de letras nítidos
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # 4. Binarización de Otsu: Blanco y negro puro (elimina el amarillo)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Configuración de Tesseract: 
        # PSM 7: Una sola línea de texto. Whitelist: Solo letras y números.
        config_tess = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        texto = pytesseract.image_to_string(binary, config=config_tess)
        return texto.strip().upper()
    except Exception as e:
        print(f"Error procesando OCR: {e}")
        return ""

def main():
    # Cargar modelo TFLite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, input_h, input_w, _ = input_details[0]['shape']

    # Cargar imagen
    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print(f"No se encontró la imagen en {IMAGE_PATH}")
        return
    h_orig, w_orig = original_img.shape[:2]

    # Pre-proceso para el modelo (Normalización y Redimensionamiento)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(rgb_img, (input_w, input_h))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    
    # Ajustar forma del output si es necesario (YOLOv8 estándar)
    if output.shape[0] == 5:
        output = output.T

    print("--- Resultados de Detección ---")
    for detection in output:
        conf = detection[4]
        if conf > CONF_THRESHOLD:
            cx, cy, w, h = detection[:4]
            
            # Cálculo de coordenadas escaladas
            if np.max(detection[:4]) <= 1.1:
                x1_raw, y1_raw = (cx - w/2) * w_orig, (cy - h/2) * h_orig
                x2_raw, y2_raw = (cx + w/2) * w_orig, (cy + h/2) * h_orig
            else:
                x1_raw = (cx - w/2) * (w_orig / input_w)
                y1_raw = (cy - h/2) * (h_orig / input_h)
                x2_raw = (cx + w/2) * (w_orig / input_w)
                y2_raw = (cy + h/2) * (h_orig / input_h)

            # Ajustar bordes con margen de seguridad (Padding de 5px)
            x1, y1 = int(max(0, x1_raw - 5)), int(max(0, y1_raw - 5))
            x2, y2 = int(min(w_orig, x2_raw + 5)), int(min(h_orig, y2_raw + 5))

            # --- PROCESO OCR ---
            placa_roi = original_img[y1:y2, x1:x2]
            
            if placa_roi.size > 0:
                texto_placa = mejorar_y_leer_placa(placa_roi)
                
                # Filtrar resultados por longitud (Placas colombianas suelen ser de 6 caracteres)
                if len(texto_placa) >= 5:
                    print(f"Placa: {texto_placa} | Confianza: {conf:.2f}")

                    # Dibujar resultados en la imagen final
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Fondo para el texto
                    cv2.rectangle(original_img, (x1, y1 - 35), (x1 + 180, y1), (0, 255, 0), -1)
                    cv2.putText(original_img, texto_placa, (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Guardar resultado
    cv2.imwrite("resultado_ocr.jpg", original_img)
    print("\nProceso completo. Revisa 'resultado_ocr.jpg'")

if __name__ == "__main__":
    main()
