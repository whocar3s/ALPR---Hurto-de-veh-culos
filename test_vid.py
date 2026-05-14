import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import time

# ============================================
# CONFIG
# ============================================

MODEL_PATH = "best_float32.tflite"

# Webcam USB
CAMERA_SOURCE = 0

# RTSP (si quieres usar IP camera)
# CAMERA_SOURCE = "rtsp://usuario:password@IP:puerto/stream"

CONF_THRESHOLD = 0.25

SAVE_FOLDER = "placas_detectadas"

# ============================================
# CREAR CARPETA
# ============================================

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ============================================
# MAIN
# ============================================

def main():

    # ====================================
    # CARGAR MODELO
    # ====================================

    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, in_h, in_w, _ = input_details[0]['shape']

    print("===================================")
    print("MODELO CARGADO")
    print("Input shape:", input_details[0]['shape'])
    print("===================================")

    # ====================================
    # CÁMARA
    # ====================================

    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():

        print("❌ No se pudo abrir cámara")
        return

    # bajar resolución (mejor para Raspberry)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # reducir delay RTSP
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("📷 Cámara iniciada")

    # ====================================
    # CONTROL GUARDADO
    # ====================================

    ultimo_guardado = 0
    contador = 0

    # ====================================
    # LOOP
    # ====================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print("⚠️ Frame perdido")
            continue

        h_orig, w_orig = frame.shape[:2]

        # ====================================
        # PREPROCESS
        # ====================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        input_data = cv2.resize(
            rgb,
            (in_w, in_h)
        )

        input_data = input_data.astype(
            np.float32
        ) / 255.0

        input_data = np.expand_dims(
            input_data,
            axis=0
        )

        # ====================================
        # INFERENCIA
        # ====================================

        interpreter.set_tensor(
            input_details[0]['index'],
            input_data
        )

        interpreter.invoke()

        output = np.squeeze(
            interpreter.get_tensor(
                output_details[0]['index']
            )
        )

        # ====================================
        # FORMATO YOLOv8
        # ====================================

        if output.shape[0] < output.shape[1]:
            output = output.T

        detecciones = []

        max_conf = 0

        # ====================================
        # LEER DETECCIONES
        # ====================================

        for row in output:

            # YOLO exportado:
            # [x, y, w, h, conf]

            conf = row[4]

            if conf > max_conf:
                max_conf = conf

            if conf < CONF_THRESHOLD:
                continue

            cx, cy, w, h = row[:4]

            # NORMALIZADO
            if np.max(row[:4]) <= 1.01:

                x1 = int((cx - w/2) * w_orig)
                y1 = int((cy - h/2) * h_orig)
                x2 = int((cx + w/2) * w_orig)
                y2 = int((cy + h/2) * h_orig)

            # ESCALADO
            else:

                x1 = int((cx - w/2) * (w_orig / in_w))
                y1 = int((cy - h/2) * (h_orig / in_h))
                x2 = int((cx + w/2) * (w_orig / in_w))
                y2 = int((cy + h/2) * (h_orig / in_h))

            # límites
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w_orig, x2)
            y2 = min(h_orig, y2)

            detecciones.append(
                (
                    [x1, y1, x2, y2],
                    float(conf)
                )
            )

        # ====================================
        # DEBUG
        # ====================================

        cv2.putText(
            frame,
            f"Max conf: {max_conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        # ====================================
        # NO DETECCIONES
        # ====================================

        if len(detecciones) == 0:

            cv2.imshow(
                "ALPR DETECTION",
                frame
            )

            if cv2.waitKey(1) & 0xFF == 27:
                break

            continue

        # ====================================
        # MEJOR DETECCIÓN
        # ====================================

        best_box, best_conf = max(
            detecciones,
            key=lambda x: x[1]
        )

        x1, y1, x2, y2 = best_box

        # ====================================
        # ROI
        # ====================================

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # ====================================
        # DIBUJAR
        # ====================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"PLACA {best_conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        # ====================================
        # MOSTRAR ROI
        # ====================================

        cv2.imshow(
            "ROI",
            roi
        )

        # ====================================
        # GUARDAR RECORTE
        # ====================================

        tiempo_actual = time.time()

        # guardar máximo 1 cada 2 segundos
        if tiempo_actual - ultimo_guardado > 2:

            nombre = (
                f"{SAVE_FOLDER}/"
                f"placa_{contador}.jpg"
            )

            cv2.imwrite(
                nombre,
                roi
            )

            print("\n========================")
            print("🚗 PLACA DETECTADA")
            print("Confianza:", round(best_conf, 3))
            print("Guardada:", nombre)
            print("========================\n")

            ultimo_guardado = tiempo_actual
            contador += 1

        # ====================================
        # MOSTRAR FRAME
        # ====================================

        cv2.imshow(
            "ALPR DETECTION",
            frame
        )

        # ESC = salir
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ====================================
    # RELEASE
    # ====================================

    cap.release()
    cv2.destroyAllWindows()

# ============================================
# START
# ============================================

if __name__ == "__main__":
    main()
