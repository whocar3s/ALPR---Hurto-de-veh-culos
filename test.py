import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import re

# --- CONFIGURACIÓN ---
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "tu_foto_de_placa.jpg"  # Cambia por tu archivo
CONF_THRESHOLD = 0.35

def validar_formato_colombiano(texto):
    """
    Aplica reglas de negocio para placas colombianas:
    1. Primeros 3 caracteres: Siempre letras.
    2. Caracteres 4 y 5: Siempre números.
    3. Carácter 6: Número (Carro) o Letra (Moto).
    """
    # Mapas de corrección por similitud visual
    a_letras = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '7': 'T'}
    a_numeros = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'T': '7', 'G': '6'}

    # Limpiar caracteres no deseados
    chars = list(re.sub(r"[^A-Z0-9]", "", texto.upper()))
    if len(chars) < 5: return "".join(chars)

    # REGLA 1: Los 3 primeros son letras (Índices 0, 1, 2)
    for i in range(min(3, len(chars))):
        if chars[i] in a_letras:
            chars[i] = a_letras[chars[i]]

    # REGLA 2: Los caracteres 4 y 5 son números (Índices 3, 4)
    for i in range(3, min(5, len(chars))):
        if chars[i] in a_numeros:
            chars[i] = a_numeros[chars[i]]

    # REGLA 3: El 6to carácter (Índice 5) - Dejamos que el OCR decida,
    # pero ya protegimos los números anteriores que es donde más falla.
    
    return "".join(chars)

def mejorar_y_leer_placa(roi):
    """Procesamiento de imagen quirúrgico para Tesseract"""
    try:
        # 1. Zoom x4 (Vital para ver la letra pequeña de las motos)
        roi = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        # 2. Pasar a Grises
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Filtro de Nitidez (Sharpening) para separar caracteres pegados
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # 4. Binarización de Otsu (Blanco y Negro puro)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Configuración OCR: PSM 7 (una línea) y OEM 3 (Default)
        # No usamos whitelist aquí para dejar que la lógica de formato corrija después
        config_tess = "--psm 7 --oem 3"
        texto_raw = pytesseract.image_to_string(binary, config=config_tess)
        
        return validar_formato_colombiano(texto_raw.strip())
    except Exception as e:
        print(f"Error en OCR: {e}")
        return ""

def main():
    # 1. Cargar Modelo TFLite
    print("Cargando modelo...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, input_h, input_w, _ = input_details[0]['shape']

    # 2. Cargar Imagen
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Imagen no encontrada.")
        return
    h_orig, w_orig = img.shape[:2]

    # 3. Inferencia TFLite
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(rgb, (input_w, input_h))
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    if output.shape[0] == 5: output = output.T

    # 4. Procesar Detecciones
    print("Analizando imagen...")
    for det in output:
        conf = det[4]
        if conf > CONF_THRESHOLD:
            cx, cy, w, h = det[:4]
            
            # Escalado de coordenadas
            if np.max(det[:4]) <= 1.1: # Normalizadas
                x1_raw, y1_raw = (cx - w/2) * w_orig, (cy - h/2) * h_orig
                x2_raw, y2_raw = (cx + w/2) * w_orig, (cy + h/2) * h_orig
            else: # Píxeles
                x1_raw = (cx - w/2) * (w_orig / input_w)
                y1_raw = (cy - h/2) * (h_orig / input_h)
                x2_raw = (cx + w/2) * (w_orig / input_w)
                y2_raw = (cy + h/2) * (h_orig / input_h)

            # Padding inteligente (Agregamos más margen a la derecha para la letra de moto)
            x1, y1 = int(max(0, x1_raw - 8)), int(max(0, y1_raw - 5))
            x2, y2 = int(min(w_orig, x2_raw + 15)), int(min(h_orig, y2_raw + 5))

            # --- OCR ---
            placa_roi = img[y1:y2, x1:x2]
            if placa_roi.size > 0:
                texto_final = mejorar_y_leer_placa(placa_roi)
                
                if len(texto_final) >= 5:
                    print(f"✅ PLACA DETECTADA: {texto_final} (Conf: {conf:.2f})")
                    
                    # Dibujo de resultados
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"ID: {texto_final}"
                    cv2.rectangle(img, (x1, y1 - 35), (x1 + 220, y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # 5. Guardar Resultado
    cv2.imwrite("resultado_final_alpr.jpg", img)
    print("\nProceso terminado. Imagen guardada como 'resultado_final_alpr.jpg'")

if __name__ == "__main__":
    main()
