import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import requests
import json
import re
import time
import os
import psutil

from datetime import datetime

# ============================================
# CONFIGURACIÓN
# ============================================

MODEL_PATH = "best_float32.tflite"

JSON_PATH = "placas_robadas.json"

API_URL = "https://apizpp.onrender.com"

CONF_THRESHOLD = 0.25

CAPTURE_INTERVAL = 20

CAMERA_SOURCE = 0

SAVE_FOLDER = "placas_detectadas"

# ============================================
# CREAR CARPETA
# ============================================

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ============================================
# CONFIG OCR
# ============================================

config = (
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# ============================================
# PLACAS YA REPORTADAS
# ============================================

placas_reportadas = set()

# ============================================
# LIMPIAR TEXTO
# ============================================

def limpiar(texto):

    texto = texto.strip()

    texto = texto.replace(" ", "")
    texto = texto.replace("\n", "")
    texto = texto.replace("\x0c", "")

    return texto

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
# MÉTRICAS SISTEMA
# ============================================

def obtener_metricas():

    cpu = psutil.cpu_percent(interval=None)

    ram = psutil.virtual_memory().percent

    temp = obtener_temperatura()

    return cpu, ram, temp

# ============================================
# CORREGIR FORMATO
# ============================================

def corregir_formato(texto, es_moto=False):

    texto = re.sub(
        r'[^A-Z0-9]',
        '',
        texto.upper()
    )

    texto = list(texto)

    if len(texto) < 6:
        return "".join(texto)

    # PRIMEROS 3 -> LETRAS
    for i in range(3):

        if texto[i] == "0":
            texto[i] = "O"

        elif texto[i] == "1":
            texto[i] = "I"

        elif texto[i] == "2":
            texto[i] = "Z"

        elif texto[i] == "5":
            texto[i] = "S"

        elif texto[i] == "8":
            texto[i] = "B"

    # POSICIONES 4 Y 5 -> NÚMEROS
    for i in [3,4]:

        if texto[i] == "O":
            texto[i] = "0"

        elif texto[i] == "I":
            texto[i] = "1"

        elif texto[i] == "Z":
            texto[i] = "2"

        elif texto[i] == "S":
            texto[i] = "5"

        elif texto[i] == "B":
            texto[i] = "8"

    # ÚLTIMO CARÁCTER
    if not es_moto:

        if texto[5] == "O":
            texto[5] = "0"

        elif texto[5] == "I":
            texto[5] = "1"

        elif texto[5] == "Z":
            texto[5] = "2"

        elif texto[5] == "S":
            texto[5] = "5"

        elif texto[5] == "B":
            texto[5] = "8"

    else:

        if texto[5] == "0":
            texto[5] = "O"

        elif texto[5] == "1":
            texto[5] = "I"

        elif texto[5] == "2":
            texto[5] = "Z"

        elif texto[5] == "5":
            texto[5] = "S"

        elif texto[5] == "8":
            texto[5] = "B"

    return "".join(texto)

# ============================================
# VALIDAR PLACA
# ============================================

def validar_placa(texto, es_moto=False):

    texto = re.sub(
        r'[^A-Z0-9]',
        '',
        texto.upper()
    )

    if len(texto) != 6:
        return None

    patron_carro = r'^[A-Z]{3}[0-9]{3}$'

    patron_moto  = r'^[A-Z]{3}[0-9]{2}[A-I]$'

    if es_moto:

        if re.match(patron_moto, texto):
            return texto

    else:

        if re.match(patron_carro, texto):
            return texto

    return None

# ============================================
# CARGAR JSON
# ============================================

def cargar_placas():

    try:

        with open(JSON_PATH, "r") as f:

            placas = json.load(f)

        return set(placas)

    except Exception as e:

        print("Error JSON:", e)

        return set()

# ============================================
# API
# ============================================

def verificar_api(placa):

    try:

        inicio_api = time.time()

        data = {

            "placa": placa,

            "ubicacion": "Raspberry Pi - Camara 1",

            "tipo_evento": "deteccion_automatica"

        }

        response = requests.post(

            f"{API_URL}/placas/verificar",

            json=data,

            timeout=5

        )

        fin_api = time.time()

        tiempo_api = (
            fin_api - inicio_api
        ) * 1000

        return response.json(), tiempo_api

    except Exception as e:

        print("Error API:", e)

        return None, -1

# ============================================
# CORREGIR PERSPECTIVA
# ============================================

def corregir_perspectiva(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower_yellow = np.array([
        15,
        40,
        40
    ])

    upper_yellow = np.array([
        40,
        255,
        255
    ])

    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    kernel = np.ones(
        (2,2),
        np.uint8
    )

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
        iterations=3
    )

    ys, xs = np.where(mask == 255)

    if len(xs) == 0:

        return None

    points = np.column_stack(
        (xs, ys)
    )

    top_left = points[
        np.argmin(
            points[:,0] + points[:,1]
        )
    ]

    top_right = points[
        np.argmax(
            points[:,0] - points[:,1]
        )
    ]

    bottom_left = points[
        np.argmin(
            points[:,0] - points[:,1]
        )
    ]

    bottom_right = points[
        np.argmax(
            points[:,0] + points[:,1]
        )
    ]

    pts1 = np.float32([

        top_left - 5,

        top_right - 5,

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

    dst = cv2.warpPerspective(

        img,

        M,

        (width,height)

    )

    return dst

# ============================================
# FILTROS OCR
# ============================================

def generar_filtros(img):

    filtros = {}

    filtros["original"] = img

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    filtros["gray"] = gray

    resize = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2
    )

    filtros["resize"] = resize

    blur = cv2.GaussianBlur(
        resize,
        (5,5),
        0
    )

    filtros["blur"] = blur

    _, thresh = cv2.threshold(
        blur,
        150,
        255,
        cv2.THRESH_BINARY
    )

    filtros["threshold"] = thresh

    adapt = cv2.adaptiveThreshold(
        resize,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    filtros["adaptive"] = adapt

    invert = cv2.bitwise_not(
        adapt
    )

    filtros["invert"] = invert

    return filtros

# ============================================
# MAIN
# ============================================

def main():

    placas_robadas = cargar_placas()

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

    cap = cv2.VideoCapture(
        CAMERA_SOURCE
    )

    if not cap.isOpened():

        print("❌ No se pudo abrir cámara")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print("📷 Cámara iniciada")

    # ====================================
    # LOOP
    # ====================================

    while True:

        inicio_total = time.time()

        # ====================================
        # CAPTURA
        # ====================================

        inicio_captura = time.time()

        ret, frame = cap.read()

        fin_captura = time.time()

        tiempo_captura = (
            fin_captura - inicio_captura
        ) * 1000

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
        # YOLO
        # ====================================

        inicio_yolo = time.time()

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

        fin_yolo = time.time()

        tiempo_yolo = (
            fin_yolo - inicio_yolo
        ) * 1000

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

            # IMPORTANTE
            # ESTE ERA EL PROCESAMIENTO
            # QUE FUNCIONABA

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

                x1 = int(
                    (cx - w/2)
                    * (w_orig / in_w)
                )

                y1 = int(
                    (cy - h/2)
                    * (h_orig / in_h)
                )

                x2 = int(
                    (cx + w/2)
                    * (w_orig / in_w)
                )

                y2 = int(
                    (cy + h/2)
                    * (h_orig / in_h)
                )

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

        print(
            f"Max confidence: {max_conf:.3f}"
        )

        # ====================================
        # SIN DETECCIONES
        # ====================================

        if len(detecciones) == 0:

            print("No se detectaron placas")

            time.sleep(CAPTURE_INTERVAL)

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

            print("ROI vacía")

            time.sleep(CAPTURE_INTERVAL)

            continue

        # ====================================
        # CLASIFICAR VEHÍCULO
        # ====================================

        ancho = x2 - x1

        largo = y2 - y1

        es_moto = not (
            ancho > (largo * 2)
        )

        # ====================================
        # OCR
        # ====================================

        inicio_ocr = time.time()

        placa = corregir_perspectiva(
            roi
        )

        if placa is None:

            print(
                "No se pudo corregir perspectiva"
            )

            time.sleep(CAPTURE_INTERVAL)

            continue

        filtros = generar_filtros(
            placa
        )

        placas_detectadas = []

        for nombre_filtro, imagen_proc in filtros.items():

            texto = pytesseract.image_to_string(

                imagen_proc,

                config=config

            )

            texto = limpiar(texto)

            texto = corregir_formato(
                texto,
                es_moto
            )

            print(
                f"{nombre_filtro}: {texto}"
            )

            if len(texto) == 6:

                placas_detectadas.append(
                    texto
                )

        fin_ocr = time.time()

        tiempo_ocr = (
            fin_ocr - inicio_ocr
        ) * 1000

        # ====================================
        # PLACA MÁS REPETIDA
        # ====================================

        mejor_valido = None

        if len(placas_detectadas) > 0:

            conteo = {}

            for placa_detectada in placas_detectadas:

                if placa_detectada not in conteo:

                    conteo[
                        placa_detectada
                    ] = 0

                conteo[
                    placa_detectada
                ] += 1

            candidata = max(
                conteo,
                key=conteo.get
            )

            placa_validada = validar_placa(
                candidata,
                es_moto
            )

            if placa_validada is not None:

                mejor_valido = placa_validada

        # ====================================
        # MÉTRICAS
        # ====================================

        fin_total = time.time()

        tiempo_total = (
            fin_total - inicio_total
        ) * 1000

        fps = 1 / (
            tiempo_total / 1000
        )

        cpu, ram, temp = obtener_metricas()

        print("\n========================")

        print("PLACA:", mejor_valido)

        print(f"Confianza YOLO: {best_conf:.3f}")

        print(f"Captura: {tiempo_captura:.2f} ms")

        print(f"YOLO: {tiempo_yolo:.2f} ms")

        print(f"OCR: {tiempo_ocr:.2f} ms")

        print(f"Pipeline total: {tiempo_total:.2f} ms")

        print(f"FPS: {fps:.2f}")

        print(f"CPU: {cpu:.2f}%")

        print(f"RAM: {ram:.2f}%")

        print(f"Temperatura: {temp:.2f} °C")

        # ====================================
        # VALIDACIÓN LOCAL
        # ====================================

        if mejor_valido:

            # YA REPORTADA
            if mejor_valido in placas_reportadas:

                print(
                    "Placa ya reportada anteriormente"
                )

            else:

                # ROBADA
                if mejor_valido in placas_robadas:

                    print(
                        "⚠ VEHÍCULO ROBADO"
                    )

                    respuesta, tiempo_api = verificar_api(
                        mejor_valido
                    )

                    print(
                        f"API: {tiempo_api:.2f} ms"
                    )

                    print("Respuesta API:")

                    print(respuesta)

                    placas_reportadas.add(
                        mejor_valido
                    )

                else:

                    print(
                        "Vehículo no reportado"
                    )

        else:

            print(
                "No se obtuvo placa válida"
            )

        print("========================\n")

        # ====================================
        # GUARDAR SOLO 1 RECORTE
        # ====================================

        if mejor_valido:

            nombre = (
                f"{SAVE_FOLDER}/"
                f"{mejor_valido}.jpg"
            )

            if not os.path.exists(nombre):

                cv2.imwrite(
                    nombre,
                    roi
                )

                print(
                    f"Recorte guardado: {nombre}"
                )

        # ====================================
        # ESPERA
        # ====================================

        time.sleep(CAPTURE_INTERVAL)

    # ====================================
    # RELEASE
    # ====================================

    cap.release()

# ============================================
# START
# ============================================

if __name__ == "__main__":

    main()
