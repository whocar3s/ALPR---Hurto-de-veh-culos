import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 1. Configuración
MODEL_PATH = "best_float32.tflite" # Verifica que el nombre sea exacto
IMAGE_PATH = "tu_foto_de_placa.jpg" 
CONF_THRESHOLD = 0.25 

# 2. Cargar el modelo
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# 3. Cargar y preparar la imagen
original_img = cv2.imread(IMAGE_PATH)
if original_img is None:
    print(f"Error: No se encontró la imagen en {IMAGE_PATH}")
    exit()

h_orig, w_orig = original_img.shape[:2]

# AJUSTE PARA FLOAT32: RGB y Normalización 0-1 estricta
rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
input_img = cv2.resize(rgb_img, (width, height))
input_img = input_img.astype(np.float32) / 255.0  # Obligatorio para float32
input_img = np.expand_dims(input_img, axis=0)

# 4. Inferencia
interpreter.set_tensor(input_details[0]['index'], input_img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# 5. Post-procesamiento
output = np.squeeze(output)

# YOLOv8 suele entregar [5, 2100]. Si es así, transponemos.
if output.shape[0] == 5:
    output = output.T

# --- DIAGNÓSTICO EN CONSOLA ---
max_conf = np.max(output[:, 4])
print(f"Modelo cargado: {MODEL_PATH}")
print(f"Confianza más alta encontrada: {max_conf:.4f}")

if max_conf < CONF_THRESHOLD:
    print("AVISO: No se alcanzó el umbral. Prueba bajando CONF_THRESHOLD.")
# ------------------------------

for detection in output:
    conf = detection[4]
    if conf > CONF_THRESHOLD:
        cx, cy, w, h = detection[:4]
        
        # Escalar coordenadas a la imagen original
        # Importante: cx, cy, w, h suelen venir en escala del modelo (0 a 320)
        x1 = int((cx - w/2) * (w_orig / width))
        y1 = int((cy - h/2) * (h_orig / height))
        x2 = int((cx + w/2) * (w_orig / width))
        y2 = int((cy + h/2) * (h_orig / height))
        
        # Dibujar
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(original_img, f"Placa: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite("resultado_float32.jpg", original_img)
print("Imagen guardada como 'resultado_float32.jpg'")
