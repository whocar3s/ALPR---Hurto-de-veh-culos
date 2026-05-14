import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import os
import time
import re
import psutil

# ============================================
# CONFIG
# ============================================

MODEL_PATH = "best_float32.tflite"

# Webcam USB
CAMERA_SOURCE = 0

# RTSP
# CAMERA_SOURCE = "rtsp://usuario:password@IP:puerto/stream"

CONF_THRESHOLD = 0.25

SAVE_FOLDER = "placas_detectadas"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ============================================
# OCR CONFIG
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
# TEMPERATURA CPU
# ============================================

def obtener_temperatura():

    try:

        with open(
            "/sys/class/thermal/thermal_zone0/temp",
            "r"
        ) as f:

            temp = float(f.read()) / 1000.0

        return round(temp, 2)

    except:

        return -1

# ============================================
# USO RAM
# ============================================

def obtener_ram():

    ram = psutil.virtual_memory()

    usado = ram.used / (1024**2)
    total = ram.total / (1024**2)

    return round(usado, 1), round(total, 1)

# ============================================
# CORREGIR PERSPECTIVA
# ============================================

def corregir_perspectiva(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    kernel = np.ones((2,2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    ys, xs = np.where(mask == 255)

    if len(xs) == 0:
        return None, mask

    points = np.column_stack((xs, ys))

    top_left = points[np.argmin(points[:,0] + points[:,1])]
    top_right = points[np.argmax(points[:,0] - points[:,1])]
    bottom_left = points[np.argmin(points[:,0] - points[:,1])]
    bottom_right = points[np.argmax(points[:,0] + points[:,1])]

    pts1 = np.float32([
        top_left,
        top_right,
        bottom_left,
        bottom_right
    ])

    width = 300
    height = 100

    pts2 = np.float32([
        [0,0],
        [width,0],
        [0,height],
        [width,height]
    ])

    M = cv2.getPerspectiveTransform(
        pts1,
        pts2
    )

    placa = cv2.warpPerspective(
        img,
        M,
        (width, height)
    )

    return placa, mask

# ============================================
# FILTROS OCR
# SOLO:
# gray
# resize
# blur
# ============================================

def generar_filtros(img):

    filtros = {}

    # ====================================
    # GRAY
    # ====================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    filtros["gray"] = gray

    # ====================================
    # RESIZE
    # ====================================

    resize = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3
    )

    filtros["resize"] = resize

    # ====================================
    # BLUR
    # ====================================

    blur = cv2.GaussianBlur(
        resize,
        (5,5),
        0
    )

    filtros["blur"] = blur

    return filtros

# ============================================
# OCR MULTIFILTRO
# ============================================

def leer_placa(roi):

    filtros = generar_filtros(roi)

    resultados = []

    for nombre, imagen in filtros.items():

        texto = pytesseract.image_to_string(
            imagen,
            config=OCR_CONFIG
        )

        texto = limpiar_texto(texto)

        print(f"[{nombre}] -> {texto}")

        if len(texto) >= 5:

            resultados.append(texto)

    if len(resultados) == 0:
        return "", filtros

    mejor = max(
        resultados,
        key=len
    )

    return mejor, filtros

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

    print("\n===================================")
    print("MODELO CARGADO")
    print("Input:", input_details[0]['shape'])
    print("===================================\n")

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

    print("📷 Cámara iniciada\n")

    ultimo_guardado = 0
    contador = 0

    # ====================================
    # LOOP
    # ====================================

    while True:

        # ====================================
        # TIEMPO CAPTURA
        # ====================================

        t0 = time.time()

        ret, frame = cap.read()

        captura_time = (
            time.time() - t0
        )

        if not ret:

            print("Frame perdido")
            continue

        h_orig, w_orig = frame.shape[:2]

        # ====================================
        # DETECCIÓN
        # ====================================

        t1 = time.time()

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

        detect_time = (
            time.time() - t1
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

            if np.max(row[:4]) <= 1.01:

                x1 = int((cx - w/2) * w_orig)
                y1 = int((cy - h/2) * h_orig)
                x2 = int((cx + w/2) * w_orig)
                y2 = int((cy + h/2) * h_orig)

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
        # SIN DETECCIONES
        # ====================================

        if len(detecciones) == 0:

            print(
                f"Sin detección | Max conf: {max_conf:.3f}",
                end="\r"
            )

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
        # PERSPECTIVA
        # ====================================

        placa, mask = corregir_perspectiva(
            roi
        )

        if placa is None:

            placa = roi

        # ====================================
        # OCR
        # ====================================

        t2 = time.time()

        texto, filtros = leer_placa(
            placa
        )

        ocr_time = (
            time.time() - t2
        )

        placa_valida = validar_placa(
            texto
        )

        # ====================================
        # RECURSOS
        # ====================================

        ram_used, ram_total = obtener_ram()

        temperatura = obtener_temperatura()

        # ====================================
        # GUARDAR
        # ====================================

        tiempo_actual = time.time()

        if tiempo_actual - ultimo_guardado > 2:

            roi_name = (
                f"{SAVE_FOLDER}/"
                f"placa_{contador}.jpg"
            )

            cv2.imwrite(
                roi_name,
                roi
            )

            cv2.imwrite(
                f"{SAVE_FOLDER}/{contador}_placa.jpg",
                placa
            )

            cv2.imwrite(
                f"{SAVE_FOLDER}/{contador}_mask.jpg",
                mask
            )

            for nombre_filtro, imagen_filtro in filtros.items():

                cv2.imwrite(
                    f"{SAVE_FOLDER}/{contador}_{nombre_filtro}.jpg",
                    imagen_filtro
                )

            # ====================================
            # CONSOLA
            # ====================================

            print("\n===================================")
            print("PLACA DETECTADA")
            print("OCR:", texto)
            print("VALIDA:", placa_valida)
            print("CONF:", round(best_conf, 3))
            print("-----------------------------------")
            print("CAPTURA :", round(captura_time, 3), "s")
            print("DETECCION:", round(detect_time, 3), "s")
            print("OCR      :", round(ocr_time, 3), "s")
            print("-----------------------------------")
            print("🌡 TEMP CPU :", temperatura, "°C")
            print(
                "RAM      :",
                ram_used,
                "/",
                ram_total,
                "MB"
            )
            print("-----------------------------------")
            print("ROI:", roi_name)
            print("===================================\n")

            ultimo_guardado = tiempo_actual
            contador += 1

# ============================================
# START
# ============================================

if __name__ == "__main__":
    main()
