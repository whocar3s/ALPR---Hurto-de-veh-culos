import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 1. Configuración
MODEL_PATH = "best_int8.tflite"
IMAGE_PATH = "tu_foto_de_placa.jpg" # Cambia esto por el nombre de tu imagen
CONF_THRESHOLD = 0.25 # Umbral de confianza

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

# Redimensionar y normalizar
input_img = cv2.resize(original_img, (width, height))
input_img = input_img.astype(np.float32) / 255.0
input_img = np.expand_dims(input_img, axis=0)

# 4. Inferencia
interpreter.set_tensor(input_details[0]['index'], input_img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# 5. Post-procesamiento (YOLOv8 entrega [1, 5, 2100])
output = np.squeeze(output).T

for detection in output:
    conf = detection[4]
    if conf > CONF_THRESHOLD:
        # Extraer centro x, centro y, ancho, alto
        cx, cy, w, h = detection[:4]
        
        # Escalar a las dimensiones originales de la imagen
        # Multiplicamos por la escala: (original / 320)
        x1 = int((cx - w/2) * (w_orig / width))
        y1 = int((cy - h/2) * (h_orig / height))
        x2 = int((cx + w/2) * (w_orig / width))
        y2 = int((cy + h/2) * (h_orig / height))
        
        # Dibujar el cuadro
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(original_img, f"Placa: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 6. Guardar y mostrar
cv2.imwrite("resultado_deteccion.jpg", original_img)
print("Detección terminada. Revisa 'resultado_deteccion.jpg'")import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 1. Configuración
MODEL_PATH = "best_int8.tflite"
IMAGE_PATH = "tu_foto_de_placa.jpg" # Cambia esto por el nombre de tu imagen
CONF_THRESHOLD = 0.25 # Umbral de confianza

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

# Redimensionar y normalizar
input_img = cv2.resize(original_img, (width, height))
input_img = input_img.astype(np.float32) / 255.0
input_img = np.expand_dims(input_img, axis=0)

# 4. Inferencia
interpreter.set_tensor(input_details[0]['index'], input_img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# 5. Post-procesamiento (YOLOv8 entrega [1, 5, 2100])
output = np.squeeze(output).T

for detection in output:
    conf = detection[4]
    if conf > CONF_THRESHOLD:
        # Extraer centro x, centro y, ancho, alto
        cx, cy, w, h = detection[:4]
        
        # Escalar a las dimensiones originales de la imagen
        # Multiplicamos por la escala: (original / 320)
        x1 = int((cx - w/2) * (w_orig / width))
        y1 = int((cy - h/2) * (h_orig / height))
        x2 = int((cx + w/2) * (w_orig / width))
        y2 = int((cy + h/2) * (h_orig / height))
        
        # Dibujar el cuadro
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(original_img, f"Placa: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 6. Guardar y mostrar
cv2.imwrite("resultado_deteccion.jpg", original_img)
print("Detección terminada. Revisa 'resultado_deteccion.jpg'")
