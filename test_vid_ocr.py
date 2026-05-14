import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import os
import time
import re

# ============================================
# CONFIG
# ============================================

MODEL_PATH = "best_float32.tflite"

# Webcam
CAMERA_SOURCE = 0

# RTSP
# CAMERA_SOURCE = "rtsp://usuario:password@IP:puerto/stream"

CONF_THRESHOLD = 0.25

SAVE_FOLDER = "placas_detectadas"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ============================================
# CONFIG OCR
# ============================================

OCR_CONFIG = (
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# ============================================
# LIMPIAR TEXTO
# ============================================

def limpiar_texto(texto):

    texto = texto.upper()

    texto = texto.replace(" ", "")
    texto = texto.replace("\n", "")
    texto = texto.replace("\x0c", "")

    texto = re.sub(
        r'[^A-Z0-9]',
        '',
        texto
    )

    return texto

# ============================================
# VALIDAR PLACA
# ============================================

def validar_placa(texto):

    patron = r'^[A-Z]{3}[0-9]{3}$'

    if re.match(patron, texto):
        return True

    return False

# ============================================
# PREPROCESAMIENTO OCR
# ============================================

def procesar_ocr(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # resize grande ayuda muchísimo
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh

# ============================================
# OCR
# ============================================

def leer_placa(roi):

    proc = procesar_ocr(roi)

    texto = pytesseract.image_to_string(
        proc,
        config=OCR_CONFIG
    )

    texto = limpiar_texto(texto)

    return texto, proc

# ============================================
# MAIN
# ============================================

def main():

    # ====================================
    # MODELO
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

        print("No se pudo abrir cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("📷 Cámara iniciada")

    ultimo_guardado = 0
    contador = 0

    # ====================================
    # LOOP
    # ====================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Frame perdido")
            continue

        h_orig, w_orig = frame.shape[:2]

        # ====================================
        # PREPROCESS YOLO
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
        # YOLOv8
        # ====================================

        if output.shape[0] < output.shape[1]:
            output = output.T

        detecciones = []

        max_conf = 0

        # ====================================
        # DETECCIONES
        # ====================================

        for row in output:

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
                "ALPR",
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

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # ====================================
        # OCR
        # ====================================

        texto, proc = leer_placa(roi)

        # ====================================
        # VALIDAR
        # ====================================

        placa_valida = validar_placa(texto)

        # ====================================
        # COLOR
        # ====================================

        color = (
            (0,255,0)
            if placa_valida
            else (0,0,255)
        )

        # ====================================
        # DIBUJAR
        # ====================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        texto_mostrar = (
            texto
            if texto != ""
            else "LEYENDO..."
        )

        cv2.putText(
            frame,
            texto_mostrar,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        # ====================================
        # MOSTRAR DEBUG
        # ====================================

        cv2.imshow(
            "ROI",
            roi
        )

        cv2.imshow(
            "OCR",
            proc
        )

        # ====================================
        # GUARDAR DETECCIONES
        # ====================================

        tiempo_actual = time.time()

        if placa_valida:

            if tiempo_actual - ultimo_guardado > 2:

                nombre = (
                    f"{SAVE_FOLDER}/"
                    f"{texto}_{contador}.jpg"
                )

                cv2.imwrite(
                    nombre,
                    roi
                )

                print("\n======================")
                print("PLACA DETECTADA")
                print("Placa:", texto)
                print("Confianza:", round(best_conf, 3))
                print("Guardada:", nombre)
                print("======================\n")

                ultimo_guardado = tiempo_actual
                contador += 1

        # ====================================
        # MOSTRAR FRAME
        # ====================================

        cv2.imshow(
            "ALPR",
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
