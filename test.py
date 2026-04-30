import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import re

# --- CONFIGURACIÓN ---
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "multiples_placas.jpg" 
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45  # Umbral para decidir si dos cuadros son la misma placa

def validar_formato_colombiano(texto):
    a_letras = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '7': 'T'}
    a_numeros = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'B': '8', 'T': '7', 'G': '6'}
    chars = list(re.sub(r"[^A-Z0-9]", "", texto.upper()))
    if len(chars) < 5: return "".join(chars)
    for i in range(min(3, len(chars))):
        if chars[i] in a_letras: chars[i] = a_letras[chars[i]]
    for i in range(3, min(5, len(chars))):
        if chars[i] in a_numeros: chars[i] = a_numeros[chars[i]]
    return "".join(chars)

def mejorar_y_leer_placa(roi):
    try:
        roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config_tess = "--psm 7"
        texto_raw = pytesseract.image_to_string(binary, config=config_tess)
        return validar_formato_colombiano(texto_raw.strip())
    except: return ""

def main():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, input_h, input_w, _ = input_details[0]['shape']

    img = cv2.imread(IMAGE_PATH)
    if img is None: return
    h_orig, w_orig = img.shape[:2]

    # Inferencia
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(rgb, (input_w, input_h))
    input_data = input_data.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    if output.shape[0] == 5: output = output.T

    # --- NUEVO: FILTRADO NMS ---
    boxes, confidences = [], []
    for det in output:
        conf = det[4]
        if conf > CONF_THRESHOLD:
            cx, cy, w, h = det[:4]
            # Convertir a formato [x1, y1, w, h] para el NMS de OpenCV
            x = int((cx - w/2) * (w_orig / input_w))
            y = int((cy - h/2) * (h_orig / input_h))
            width = int(w * (w_orig / input_w))
            height = int(h * (h_orig / input_h))
            boxes.append([x, y, width, height])
            confidences.append(float(conf))

    # Aplicar Non-Maximum Suppression (Elimina cuadros duplicados sobre la misma placa)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    if len(indices) > 0:
        print(f"Se encontraron {len(indices)} placas.")
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            conf = confidences[i]

            # Padding
            x1, y1 = max(0, x - 5), max(0, y - 5)
            x2, y2 = min(w_orig, x + w + 10), min(h_orig, y + h + 5)

            placa_roi = img[y1:y2, x1:x2]
            if placa_roi.size > 0:
                texto = mejorar_y_leer_placa(placa_roi)
                if len(texto) >= 5:
                    print(f"Placa {i+1}: {texto}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite("resultado_multi_placa.jpg", img)

if __name__ == "__main__":
    main()
