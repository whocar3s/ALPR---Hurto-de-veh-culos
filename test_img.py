import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# --- 1. CONFIGURACIÓN ---
MODEL_PATH = "best_float32.tflite"  # Tu modelo float32
IMAGE_PATH = "tu_foto_de_placa.jpg"   # Nombre de tu imagen de prueba
CONF_THRESHOLD = 0.30                # Umbral de confianza

def main():
    # --- 2. CARGAR EL MODELO ---
    print(f"Cargando modelo: {MODEL_PATH}...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Obtener dimensiones que espera el modelo (ej. 320x320)
    _, input_h, input_w, _ = input_details[0]['shape']

    # --- 3. PREPARAR LA IMAGEN ---
    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print(f"Error: No se pudo cargar la imagen {IMAGE_PATH}")
        return

    h_orig, w_orig = original_img.shape[:2]

    # YOLOv8 prefiere RGB. Convertimos y redimensionamos.
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(rgb_img, (input_w, input_h))

    # Normalización para Float32 (0 a 1)
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # --- 4. INFERENCIA ---
    print("Ejecutando detección...")
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # --- 5. POST-PROCESAMIENTO ---
    # Eliminar dimensiones innecesarias y transponer de [5, 2100] a [2100, 5]
    output = np.squeeze(output)
    if output.shape[0] == 5:
        output = output.T

    max_conf_encontrada = np.max(output[:, 4])
    print(f"Confianza más alta detectada: {max_conf_encontrada:.4f}")

    detecciones_reales = 0
    for detection in output:
        conf = detection[4]
        
        if conf > CONF_THRESHOLD:
            detecciones_reales += 1
            cx, cy, w, h = detection[:4]

            # DETERMINAR ESCALADO
            # Si el valor máximo es <= 1.1, las coordenadas están normalizadas (0-1)
            # Si es mayor, vienen en píxeles del modelo (0-320)
            if np.max(detection[:4]) <= 1.1:
                x1_raw, y1_raw = (cx - w/2) * w_orig, (cy - h/2) * h_orig
                x2_raw, y2_raw = (cx + w/2) * w_orig, (cy + h/2) * h_orig
            else:
                x1_raw = (cx - w/2) * (w_orig / input_w)
                y1_raw = (cy - h/2) * (h_orig / input_h)
                x2_raw = (cx + w/2) * (w_orig / input_w)
                y2_raw = (cy + h/2) * (h_orig / input_h)

            # Convertir a enteros y asegurar que estén dentro de la imagen
            x1, y1 = int(max(0, x1_raw)), int(max(0, y1_raw))
            x2, y2 = int(min(w_orig, x2_raw)), int(min(h_orig, y2_raw))

            print(f"Placa encontrada: [{x1}, {y1}, {x2}, {y2}] Conf: {conf:.2f}")

            # --- 6. DIBUJAR ---
            # Cuadro verde grueso
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Fondo para el texto (mejora legibilidad)
            cv2.rectangle(original_img, (x1, y1 - 35), (x1 + 150, y1), (0, 255, 0), -1)
            cv2.putText(original_img, f"PLACA {conf:.2f}", (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # --- 7. GUARDAR RESULTADO ---
    if detecciones_reales > 0:
        cv2.imwrite("resultado_final.jpg", original_img)
        print(f"¡Éxito! Se detectaron {detecciones_reales} placas.")
        print("Revisa el archivo 'resultado_final.jpg'")
    else:
        print("No se dibujó nada porque ninguna detección superó el umbral.")

if __name__ == "__main__":
    main()
