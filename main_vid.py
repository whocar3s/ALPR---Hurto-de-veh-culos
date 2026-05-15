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

# ============================================
# CONFIG
# ============================================

MODEL_PATH = "best_float32.tflite"

JSON_PATH = "placas_robadas.json"

API_URL = "https://apizpp.onrender.com"

CONF_THRESHOLD = 0.25

CAPTURE_INTERVAL = 30

CAMERA_SOURCE = 0

SAVE_FOLDER = "placas_detectadas"

# ============================================
# CREAR CARPETA
# ============================================

os.makedirs(SAVE_FOLDER, exist_ok=True)

# ============================================
# OCR CONFIG
# ============================================

config = (
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# ============================================
# MAPEOS OCR
# ============================================

NUM_A_LETRA = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "8": "B"
}

LETRA_A_NUM = {
    "O": "0",
    "I": "1",
    "J": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "B": "8",
    "G": "6"
}

# ============================================
# PLACAS YA REPORTADAS
# ============================================

placas_reportadas = set()

# ============================================
# LIMPIAR Y CORREGIR OCR
# ============================================

def limpiar_y_corregir(texto):

    texto = texto.strip().upper()

    texto = "".join(texto.split())

    if not texto:
        return ""

    # ====================================
    # FILTRO RUIDO FRONTAL
    # ====================================

    if len(texto) >= 7:

        letras_iniciales = 0

        for char in texto:

            if char.isalpha():

                letras_iniciales += 1

            else:

                break

        if letras_iniciales >= 4:

            texto = texto[1:]

    # ====================================
    # RECORTE
    # ====================================

    texto = texto[:6]

    lista_texto = list(texto)

    n = len(lista_texto)

    # ====================================
    # MAPEO POSICIONAL
    # ====================================

    for i in range(n):

        # LETRAS
        if i < 3:

            if lista_texto[i].isdigit():

                lista_texto[i] = NUM_A_LETRA.get(
                    lista_texto[i],
                    lista_texto[i]
                )

        # NÚMEROS
        elif i == 3 or i == 4:

            if lista_texto[i].isalpha():

                lista_texto[i] = LETRA_A_NUM.get(
                    lista_texto[i],
                    lista_texto[i]
                )

    return "".join(lista_texto)

# ============================================
# VALIDAR PLACA
# ============================================

def validar_placa(texto):

    texto = re.sub(
        r'[^A-Z0-9]',
        '',
        texto.upper()
    )

    if len(texto) != 6:
        return None

    patron_carro = r'^[A-Z]{3}[0-9]{3}$'

    patron_moto = r'^[A-Z]{3}[0-9]{2}[A-Z]$'

    if re.match(patron_carro, texto):
        return texto

    if re.match(patron_moto, texto):
        return texto

    return None

# ============================================
# MÉTRICAS
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

def obtener_metricas():

    cpu = psutil.cpu_percent(interval=1)

    ram = psutil.virtual_memory().percent

    temp = obtener_temperatura()

    return cpu, ram, temp

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
# PERSPECTIVA AMARILLA
# ============================================

def corregir_perspectiva_amarilla(img):

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

    points = np.column_stack((xs, ys))

    top_left = points[
        np.argmin(points[:,0] + points[:,1])
    ]

    top_right = points[
        np.argmax(points[:,0] - points[:,1])
    ]

    bottom_left = points[
        np.argmin(points[:,0] - points[:,1])
    ]

    bottom_right = points[
        np.argmax(points[:,0] + points[:,1])
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

    return cv2.warpPerspective(
        img,
        M,
        (width,height)
    )

# ============================================
# PERSPECTIVA BLANCA
# ============================================

def corregir_perspectiva_blanca(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    mask_w = cv2.inRange(

        hsv,

        np.array([0,0,120]),

        np.array([180,60,255])

    )

    kernel = np.ones(
        (5,5),
        np.uint8
    )

    closed = cv2.morphologyEx(
        mask_w,
        cv2.MORPH_CLOSE,
        kernel
    )

    contornos, _ = cv2.findContours(

        closed,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    if not contornos:

        return None

    c_max = max(
        contornos,
        key=cv2.contourArea
    )

    rect = cv2.minAreaRect(c_max)

    box = cv2.boxPoints(rect)

    box = np.intp(box)

    s = box.sum(axis=1)

    diff = np.diff(box, axis=1)

    pts1 = np.zeros((4,2), dtype="float32")

    pts1[0] = box[np.argmin(s)]

    pts1[1] = box[np.argmin(diff)]

    pts1[2] = box[np.argmax(diff)]

    pts1[3] = box[np.argmax(s)]

    pts2 = np.float32([

        [0,0],

        [300,0],

        [0,100],

        [300,100]

    ])

    M = cv2.getPerspectiveTransform(
        pts1,
        pts2
    )

    return cv2.warpPerspective(
        img,
        M,
        (300,100)
    )

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

    res_lanczos = cv2.resize(

        gray,

        None,

        fx=2,

        fy=2,

        interpolation=cv2.INTER_LANCZOS4

    )

    filtros["res_lanczos"] = res_lanczos

    _, otsu = cv2.threshold(

        res_lanczos,

        0,

        255,

        cv2.THRESH_BINARY + cv2.THRESH_OTSU

    )

    filtros["otsu"] = otsu

    resize = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2
    )

    filtros["resize"] = resize

    blur = cv2.GaussianBlur(
        resize,
        (3,3),
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

    # ====================================
    # LOOP
    # ====================================

    while True:

        inicio_total = time.time()

        # ====================================
        # ABRIR CÁMARA
        # ====================================

        cap = cv2.VideoCapture(
            CAMERA_SOURCE
        )

        time.sleep(2)

        for _ in range(5):
            cap.read()

        ret, frame = cap.read()

        cap.release()

        if not ret:

            print("No se pudo capturar frame")

            time.sleep(CAPTURE_INTERVAL)

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

        for row in output:

            conf = row[4]

            if conf < CONF_THRESHOLD:
                continue

            cx, cy, w, h = row[:4]

            if np.max(row[:4]) <= 1.01:

                x1 = int((cx - w/2) * w_orig)
                y1 = int((cy - h/2) * h_orig)
                x2 = int((cx + w/2) * w_orig)
                y2 = int((cy + h/2) * h_orig)

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
                    [x1,y1,x2,y2],
                    float(conf)
                )
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

            time.sleep(CAPTURE_INTERVAL)

            continue

        # ====================================
        # MÁSCARAS
        # ====================================

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        mask_yellow = cv2.inRange(

            hsv,

            np.array([15,40,40]),

            np.array([40,255,255])

        )

        mask_white = cv2.inRange(

            hsv,

            np.array([0,0,120]),

            np.array([180,60,255])

        )

        yellow_pixels = cv2.countNonZero(
            mask_yellow
        )

        white_pixels = cv2.countNonZero(
            mask_white
        )

        # ====================================
        # PERSPECTIVA
        # ====================================

        inicio_ocr = time.time()

        if yellow_pixels > white_pixels:

            placa_img = corregir_perspectiva_amarilla(
                roi
            )

        else:

            placa_img = corregir_perspectiva_blanca(
                roi
            )

        if placa_img is None:

            print("No se pudo corregir perspectiva")

            time.sleep(CAPTURE_INTERVAL)

            continue

        # ====================================
        # OCR
        # ====================================

        filtros = generar_filtros(
            placa_img
        )

        placas_detectadas = []

        for nombre_filtro, imagen_proc in filtros.items():

            texto_raw = pytesseract.image_to_string(

                imagen_proc,

                config=config

            )

            texto_corregido = limpiar_y_corregir(
                texto_raw
            )

            if len(texto_corregido) == 6:

                placas_detectadas.append(
                    texto_corregido
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

            for placa in placas_detectadas:

                if placa not in conteo:

                    conteo[placa] = 0

                conteo[placa] += 1

            candidata = max(
                conteo,
                key=conteo.get
            )

            placa_validada = validar_placa(
                candidata
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

        print(f"YOLO: {tiempo_yolo:.2f} ms")

        print(f"OCR: {tiempo_ocr:.2f} ms")

        print(f"TOTAL: {tiempo_total:.2f} ms")

        print(f"FPS: {fps:.2f}")

        print(f"CPU: {cpu:.2f}%")

        print(f"RAM: {ram:.2f}%")

        print(f"TEMP: {temp:.2f}°C")

        # ====================================
        # VALIDACIÓN LOCAL
        # ====================================

        if mejor_valido:

            if mejor_valido in placas_reportadas:

                print(
                    "Placa ya reportada"
                )

            else:

                if mejor_valido in placas_robadas:

                    print(
                        "VEHÍCULO ROBADO"
                    )

                    respuesta, tiempo_api = verificar_api(
                        mejor_valido
                    )

                    print(
                        f"API: {tiempo_api:.2f} ms"
                    )

                    print(respuesta)

                    placas_reportadas.add(
                        mejor_valido
                    )

                else:

                    print(
                        "Vehículo no reportado"
                    )

            # ====================================
            # GUARDAR RECORTE
            # ====================================

            nombre = (
                f"{SAVE_FOLDER}/"
                f"{mejor_valido}.jpg"
            )

            if not os.path.exists(nombre):

                cv2.imwrite(
                    nombre,
                    placa_img
                )

        else:

            print(
                "No se obtuvo placa válida"
            )

        print("========================\n")

        # ====================================
        # ESPERA
        # ====================================

        time.sleep(CAPTURE_INTERVAL)

# ============================================
# START
# ============================================

if __name__ == "__main__":

    main()
