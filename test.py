import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import re

# ============================================
# CONFIGURACIÓN
# ============================================

MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "tu_foto.jpg"

CONF_THRESHOLD = 0.25

# ============================================
# CONFIG OCR
# ============================================

config = (
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

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
# CORREGIR FORMATO PLACA
# ============================================

def corregir_formato(texto, es_moto=False):

    texto = re.sub(
        r'[^A-Z0-9]',
        '',
        texto.upper()
    )

    if len(texto) != 6:
        return texto

    texto = list(texto)

    # ====================================
    # PRIMEROS 3 -> LETRAS
    # ====================================

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

    # ====================================
    # POSICIONES 4 Y 5 -> NÚMEROS
    # ====================================

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

    # ====================================
    # ÚLTIMO CARÁCTER
    # ====================================

    # CARRO -> número
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

    # MOTO -> letra
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
        iterations=3
    )

    ys, xs = np.where(mask == 255)

    if len(xs) == 0:
        return None

    points = np.column_stack((xs, ys))

    top_left = points[np.argmin(points[:,0] + points[:,1])]
    top_right = points[np.argmax(points[:,0] - points[:,1])]
    bottom_left = points[np.argmin(points[:,0] - points[:,1])]
    bottom_right = points[np.argmax(points[:,0] + points[:,1])]

    pts1 = np.float32([
        top_left - 5,
        top_right - 5,
        bottom_left,
        bottom_right
    ])

    width = 300
    height = 100

    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    M = cv2.getPerspectiveTransform(
        pts1,
        pts2
    )

    dst = cv2.warpPerspective(
        img,
        M,
        (width, height)
    )

    return dst

# ============================================
# GENERAR FILTROS OCR
# ============================================

def generar_filtros(img):

    filtros = {}

    # ORIGINAL
    filtros["original"] = img

    # GRISES
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    filtros["gray"] = gray

    # RESIZE
    resize = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2
    )

    filtros["resize"] = resize

    # BLUR
    blur = cv2.GaussianBlur(
        resize,
        (5,5),
        0
    )

    filtros["blur"] = blur

    # THRESHOLD
    _, thresh = cv2.threshold(
        blur,
        150,
        255,
        cv2.THRESH_BINARY
    )

    filtros["threshold"] = thresh

    # ADAPTIVE
    adapt = cv2.adaptiveThreshold(
        resize,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    filtros["adaptive"] = adapt

    # INVERT
    invert = cv2.bitwise_not(adapt)

    filtros["invert"] = invert

    return filtros

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

    # ====================================
    # LEER IMAGEN
    # ====================================

    img = cv2.imread(IMAGE_PATH)

    if img is None:

        print("No se pudo cargar la imagen")
        return

    h_orig, w_orig = img.shape[:2]

    # ====================================
    # PREPROCESS YOLO
    # ====================================

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    input_data = cv2.resize(
        rgb,
        (in_w, in_h)
    )

    input_data = input_data.astype(np.float32) / 255.0

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

    # YOLOv8 TFLite
    if output.shape[0] < output.shape[1]:
        output = output.T

    detecciones = []

    # ====================================
    # LEER DETECCIONES
    # ====================================

    for row in output:

        probabilidades = row[4:]

        conf = np.max(probabilidades)

        if conf > CONF_THRESHOLD:

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

            detecciones.append(
                ((x1,y1,x2,y2), conf)
            )

    # ====================================
    # SIN DETECCIONES
    # ====================================

    if len(detecciones) == 0:

        print("No se detectaron placas")

        print(
            "Conf máxima:",
            np.max(output[:,4:])
        )

        return

    # ====================================
    # MEJOR DETECCIÓN
    # ====================================

    best_box, best_conf = max(
        detecciones,
        key=lambda x: x[1]
    )

    x1, y1, x2, y2 = best_box

    # límites
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w_orig, x2)
    y2 = min(h_orig, y2)

    # ====================================
    # ROI
    # ====================================

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:

        print("ROI vacía")
        return

    # ====================================
    # CLASIFICAR VEHÍCULO
    # ====================================

    ancho = x2 - x1
    largo = y2 - y1

    es_moto = not (ancho > (largo * 2))

    # ====================================
    # PERSPECTIVA
    # ====================================

    placa = corregir_perspectiva(
        roi
    )

    if placa is None:

        print("No se pudo corregir perspectiva")
        return

    # ====================================
    # FILTROS OCR
    # ====================================

    filtros = generar_filtros(
        placa
    )

    placas_detectadas = []

    # ====================================
    # OCR POR FILTRO
    # ====================================

    for nombre_filtro, imagen_proc in filtros.items():

        texto = pytesseract.image_to_string(
            imagen_proc,
            config=config
        )

        # limpiar OCR
        texto = limpiar(texto)

        # corregir formato colombiano
        texto = corregir_formato(
            texto,
            es_moto
        )

        # validar placa
        placa_valida = validar_placa(
            texto,
            es_moto
        )

        print(f"{nombre_filtro}: {texto}")

        # guardar válidas
        if placa_valida is not None:

            placas_detectadas.append(
                placa_valida
            )

    # ====================================
    # PLACA MÁS REPETIDA
    # ====================================

    mejor_valido = None

    if len(placas_detectadas) > 0:

        conteo = {}

        for placa_detectada in placas_detectadas:

            if placa_detectada not in conteo:
                conteo[placa_detectada] = 0

            conteo[placa_detectada] += 1

        mejor_valido = max(
            conteo,
            key=conteo.get
        )

    # ====================================
    # RESULTADO FINAL
    # ====================================

    print("\n===================")

    if mejor_valido:

        print("PLACA FINAL:", mejor_valido)

    else:

        print("No se obtuvo placa válida")

    print("===================\n")

    # ====================================
    # DIBUJAR RESULTADO
    # ====================================

    texto_final = (
        mejor_valido
        if mejor_valido
        else "INVALIDA"
    )

    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    cv2.putText(
        img,
        texto_final,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # ====================================
    # GUARDAR DEBUG
    # ====================================

    cv2.imwrite(
        "debug_01_roi.jpg",
        roi
    )

    cv2.imwrite(
        "debug_02_placa.jpg",
        placa
    )

    for nombre, imagen_proc in filtros.items():

        cv2.imwrite(
            f"debug_{nombre}.jpg",
            imagen_proc
        )

    cv2.imwrite(
        "debug_final.jpg",
        img
    )

    print("Imágenes debug guardadas")

# ============================================
# START
# ============================================

if __name__ == "__main__":
    main()
