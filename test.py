import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 1. Cargar el modelo e iniciar el intérprete
interpreter = tflite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # Debería ser [1, 320, 320, 3]

def preprocess(frame, input_size):
    """Prepara la imagen de la cámara para el modelo"""
    img = cv2.resize(frame, (input_size, input_size))
    img = img.astype(np.float32) / 255.0  # Normalizar 0-1
    img = np.expand_dims(img, axis=0)     # Añadir dimensión de batch [1, 320, 320, 3]
    return img

cap = cv2.VideoCapture(0)

print("Iniciando detección ligera en Pi 3...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Pre-procesar
    input_data = preprocess(frame, input_shape[1])

    # Ejecutar Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Obtener resultados
    # YOLOv8 en TFLite suele entregar [1, 5, 2100] -> (x, y, w, h, conf)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data).T # Transponer para manejarlo mejor

    for detection in output_data:
        confidence = detection[4]
        if confidence > 0.4:  # Tu umbral de placa
            # YOLO entrega valores normalizados o en píxeles del imgsz
            # Necesitamos re-escalar a los píxeles de la cámara (640x480)
            cx, cy, w, h = detection[:4]
            
            # Convertir de centro(x,y) a esquinas(x1,y1)
            x1 = int((cx - w/2) * frame.shape[1] / input_shape[1])
            y1 = int((cy - h/2) * frame.shape[0] / input_shape[2])
            x2 = int((cx + w/2) * frame.shape[1] / input_shape[1])
            y2 = int((cy + h/2) * frame.shape[0] / input_shape[2])

            # Dibujar
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"PLACA {confidence:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("ALPR Lite Colombia", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
