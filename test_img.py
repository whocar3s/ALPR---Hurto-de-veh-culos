import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 1. Configuración
MODEL_PATH = "best_int8.tflite"
IMAGE_PATH = "tu_foto_de_placa.jpg" 
CONF_THRESHOLD = 0.15  # Lo bajamos para probar

# 2. Cargar el modelo
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

# 3. Cargar y preparar la imagen
original_img = cv2.imread(IMAGE_PATH)
if original_img is None:
    print("Error: No se encontró la imagen.")
    exit()

h_orig, w_orig = original_img.shape[:2]

# AJUSTE 1: YOLOv8 prefiere RGB (OpenCV usa BGR)
rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
input_img = cv2.resize(rgb_img, (width, height))

# AJUSTE 2: Verificar si el modelo es INT8 real
# Si el input_details['dtype'] es uint8, NO se divide por 255.0
if input_details[0]['dtype'] == np.float32:
    input_img = input_img.astype(np.float32) / 255.0
else:
    input_img = input_img.astype(np.uint8)

input_img = np.expand_dims(input_img, axis=0)

# 4. Inferencia
interpreter.set_tensor(input_details[0]['index'], input_img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# 5. Post-procesamiento (Ajustado para YOLOv8)
output = np.squeeze(output)
# YOLOv8 saca [5, 2100], necesitamos [2100, 5]
if output.shape[0] == 5:
    output = output.T

print(f"Confianza máxima encontrada: {np.max(output[:, 4]):.4f}")

for detection in output:
    conf = detection[4]
    if conf > CONF_THRESHOLD:
        cx, cy, w, h = detection[:4]
        
        # AJUSTE 3: El escalado debe ser relativo a la entrada del modelo
        # Las coordenadas cx, cy, w, h en YOLOv8 TFLite suelen venir en píxeles del modelo (0-320)
        x1 = int((cx - w/2) * (w_orig / width))
        y1 = int((cy - h/2) * (h_orig / height))
        x2 = int((cx + w/2) * (w_orig / width))
        y2 = int((cy + h/2) * (h_orig / height))
        
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(original_img, f"Placa: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite("resultado_deteccion.jpg", original_img)
print("Proceso completado. Revisa 'resultado_deteccion.jpg'")
