import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import time
import re
import statistics
import subprocess
import requests

# ============================================
# CONFIG
# ============================================

MODEL_PATH = "best_float32.tflite"

IMAGES_FOLDER = "imagenes_prueba"

CONF_THRESHOLD = 0.25

API_URL = "https://apizpp.onrender.com/placas/verificar"

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
# LIMPIAR OCR
# ============================================

def limpiar_y_corregir(texto):

    texto = texto.strip().upper()

    texto = "".join(texto.split())

    if not texto:
        return ""

    if len(texto) >= 7:

        letras_iniciales = 0

        for char in texto:

            if char.isalpha():
                letras_iniciales += 1
            else:
                break

        if letras_iniciales >= 4:
            texto = texto[1:]

    texto = texto[:6]

    lista_texto = list(texto)

    n = len(lista_texto)

    for i in range(n):

        if i < 3:

            if lista_texto[i].isdigit():

                lista_texto[i] = NUM_A_LETRA.get(
                    lista_texto[i],
                    lista_texto[i]
                )

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
# API
# ============================================

def verificar_api(placa):

    try:

        data = {

            "placa": placa,

            "ubicacion": "Prueba 30 imagenes",

            "tipo_evento": "prueba_rendimiento"

        }

        inicio_api = time.time()

        response = requests.post(

            API_URL,

            json=data,

            timeout=10

        )

        fin_api = time.time()

        tiempo_api = (
            fin_api - inicio_api
        ) * 1000

        return (
            response.json(),
            tiempo_api,
            response.status_code
        )

    except Exception as e:

        print("Error API:", e)

        return (
            None,
            None,
            None
        )

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
# CARGAR MODELO
# ============================================

interpreter = tflite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

_, in_h, in_w, _ = input_details[0]['shape']

# ============================================
# MÉTRICAS
# ============================================

tiempos_yolo = []
tiempos_ocr = []
tiempos_total = []
tiempos_api = []

cpu_usos = []
ram_usos = []
temperaturas = []

api_ok = 0
api_fail = 0

detecciones_ok = 0
detecciones_fail = 0

# ============================================
# LISTAR IMÁGENES
# ============================================

imagenes = sorted([

    f for f in os.listdir(IMAGES_FOLDER)

    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )

])

print("\n================================")
print("TOTAL IMAGENES:", len(imagenes))
print("================================\n")

# ============================================
# LOOP
# ============================================

for nombre_imagen in imagenes:

    print("\n================================")
    print("IMAGEN:", nombre_imagen)
    print("================================")

    inicio_total = time.time()

    ruta = os.path.join(
        IMAGES_FOLDER,
        nombre_imagen
    )

    img = cv2.imread(ruta)

    if img is None:

        print("Error imagen")
        continue

    h_orig, w_orig = img.shape[:2]

    # ====================================
    # PREPROCESS
    # ====================================

    rgb = cv2.cvtColor(
        img,
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

    tiempos_yolo.append(
        tiempo_yolo
    )

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

    if len(detecciones) == 0:

        print("No detecciones")

        detecciones_fail += 1

        continue

    # ====================================
    # ROI
    # ====================================

    best_box, best_conf = max(
        detecciones,
        key=lambda x: x[1]
    )

    x1, y1, x2, y2 = best_box

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:

        detecciones_fail += 1

        continue

    # ====================================
    # OCR
    # ====================================

    inicio_ocr = time.time()

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

    if yellow_pixels > white_pixels:

        placa_img = corregir_perspectiva_amarilla(
            roi
        )

    else:

        placa_img = corregir_perspectiva_blanca(
            roi
        )

    if placa_img is None:

        print("No perspectiva")

        detecciones_fail += 1

        continue

    filtros = generar_filtros(
        placa_img
    )

    placas_detectadas = []

    for _, imagen_proc in filtros.items():

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

    tiempos_ocr.append(
        tiempo_ocr
    )

    # ====================================
    # PLACA FINAL
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
    # API
    # ====================================

    if mejor_valido is not None:

        detecciones_ok += 1

        respuesta, tiempo_api, status = verificar_api(
            mejor_valido
        )

        if tiempo_api is not None:

            tiempos_api.append(
                tiempo_api
            )

        if status == 200:

            api_ok += 1

        else:

            api_fail += 1

        print(
            f"Placa: {mejor_valido}"
        )

        print(
            f"API: {tiempo_api:.2f} ms"
        )

        print(
            f"Status: {status}"
        )

    else:

        detecciones_fail += 1

        print("Placa inválida")

    # ====================================
    # TOTAL
    # ====================================

    fin_total = time.time()

    tiempo_total = (
        fin_total - inicio_total
    ) * 1000

    tiempos_total.append(
        tiempo_total
    )

    # ====================================
    # CPU
    # ====================================

    cpu = subprocess.getoutput(
        "top -bn1 | grep 'Cpu(s)'"
    )

    try:

        cpu_idle = float(
            cpu.split()[7]
            .replace(",", ".")
        )

        cpu_use = 100 - cpu_idle

    except:

        cpu_use = 0

    cpu_usos.append(cpu_use)

    # ====================================
    # RAM
    # ====================================

    ram = subprocess.getoutput(
        "free -m"
    )

    try:

        line = ram.split("\n")[1]

        usados = int(
            line.split()[2]
        )

    except:

        usados = 0

    ram_usos.append(usados)

    # ====================================
    # TEMPERATURA
    # ====================================

    try:

        temp = subprocess.getoutput(
            "vcgencmd measure_temp"
        )

        temp = float(
            temp.replace(
                "temp=",
                ""
            ).replace(
                "'C",
                ""
            )
        )

    except:

        temp = 0

    temperaturas.append(temp)

    print(
        f"YOLO: {tiempo_yolo:.2f} ms"
    )

    print(
        f"OCR: {tiempo_ocr:.2f} ms"
    )

    print(
        f"TOTAL: {tiempo_total:.2f} ms"
    )

    print(
        f"CPU: {cpu_use:.2f}%"
    )

    print(
        f"RAM: {usados} MB"
    )

    print(
        f"TEMP: {temp:.2f}°C"
    )

# ============================================
# RESULTADOS FINALES
# ============================================

print("\n================================")
print("RESULTADOS FINALES")
print("================================")

if len(tiempos_total) > 0:

    print(
        f"\nTotal imagenes: "
        f"{len(imagenes)}"
    )

    print(
        f"Detecciones correctas: "
        f"{detecciones_ok}"
    )

    print(
        f"Detecciones fallidas: "
        f"{detecciones_fail}"
    )

    accuracy = (
        detecciones_ok /
        len(imagenes)
    ) * 100

    print(
        f"Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"YOLO promedio: "
        f"{statistics.mean(tiempos_yolo):.2f} ms"
    )

    print(
        f"OCR promedio: "
        f"{statistics.mean(tiempos_ocr):.2f} ms"
    )

    print(
        f"API promedio: "
        f"{statistics.mean(tiempos_api):.2f} ms"
    )

    print(
        f"TOTAL promedio: "
        f"{statistics.mean(tiempos_total):.2f} ms"
    )

    print(
        f"CPU promedio: "
        f"{statistics.mean(cpu_usos):.2f}%"
    )

    print(
        f"RAM promedio: "
        f"{statistics.mean(ram_usos):.2f} MB"
    )

    print(
        f"TEMP promedio: "
        f"{statistics.mean(temperaturas):.2f}°C"
    )

    throughput = (
        len(tiempos_total) /
        (sum(tiempos_total) / 1000)
    )

    print(
        f"Throughput: "
        f"{throughput:.2f} img/s"
    )

    print(
        f"API exitosas: "
        f"{api_ok}"
    )

    print(
        f"API fallidas: "
        f"{api_fail}"
    )

print("\n================================\n")
