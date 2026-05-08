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
# OCR CONFIG
# ============================================

OCR_CONFIG = (
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# ============================================
# VALIDACIÓN PLACAS COLOMBIANAS
# ============================================

def validar_placa(texto, es_moto=False):

    texto = re.sub(r'[^A-Z0-9]', '', texto.upper())

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
# DETECCIÓN HSV AMARILLO
# ============================================

def detectar_placa_amarilla(roi):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return mask

# ============================================
# OBTENER ESQUINAS
# ============================================

def obtener_esquinas(mask):

    ys, xs = np.where(mask == 255)

    if len(xs) == 0:
        return None

    points = np.column_stack((xs, ys))

    top_left = points[np.argmin(points[:,0] + points[:,1])]
    top_right = points[np.argmax(points[:,0] - points[:,1])]
    bottom_left = points[np.argmin(points[:,0] - points[:,1])]
    bottom_right = points[np.argmax(points[:,0] + points[:,1])]

    return np.float32([
        top_left,
        top_right,
        bottom_left,
        bottom_right
    ])

# ============================================
# PERSPECTIVE TRANSFORM
# ============================================

def corregir_perspectiva(roi, pts1):

    width = 300
    height = 100

    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(
        roi,
        M,
        (width, height)
    )

    return dst

# ============================================
# PREPROCESSING OCR
# ============================================

def preprocess_ocr(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    blur = cv2.GaussianBlur(gray, (5,5), 0)

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
# MAIN
# ============================================

def main():

    # =========================
    # CARGAR MODELO
    # =========================

    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, in_h, in_w, _ = input_details[0]['shape']

    # =========================
    # CARGAR IMAGEN
    # =========================

    img = cv2.imread(IMAGE_PATH)

    if img is None:
        print("No se pudo cargar la imagen")
        return

    h_orig, w_orig = img.shape[:2]

    # =========================
    # PREPROCESAMIENTO YOLO
    # =========================

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = cv2.resize(
        rgb,
        (in_w, in_h)
    )

    input_data = input_data.astype(np.float32) / 255.0

    input_data = np.expand_dims(
        input_data,
        axis=0
    )

    # =========================
    # INFERENCIA
    # =========================

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

    # YOLOv8
    if output.shape[0] < output.shape[1]:
        output = output.T

    detecciones = []

    # =========================
    # LEER DETECCIONES
    # =========================

    for row in output:

        conf = np.max(row[4:])

        if conf > CONF_THRESHOLD:

            cx, cy, w, h = row[:4]

            # coordenadas
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

            detecciones.append(
                ((x1,y1,x2,y2), conf)
            )

    # =========================
    # NO DETECCIONES
    # =========================

    if len(detecciones) == 0:

        print("No se detectaron placas")
        return

    # =========================
    # MEJOR DETECCIÓN
    # =========================

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

    # =========================
    # RECORTE
    # =========================

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        print("ROI vacía")
        return

    # =========================
    # DETECTAR AMARILLO
    # =========================

    mask = detectar_placa_amarilla(roi)

    # =========================
    # OBTENER ESQUINAS
    # =========================

    pts1 = obtener_esquinas(mask)

    if pts1 is None:
        print("No se encontraron esquinas")
        return

    # =========================
    # CORREGIR PERSPECTIVA
    # =========================

    placa_recta = corregir_perspectiva(
        roi,
        pts1
    )

    # =========================
    # PREPROCESS OCR
    # =========================

    ocr_img = preprocess_ocr(
        placa_recta
    )

    # =========================
    # OCR
    # =========================

    texto = pytesseract.image_to_string(
        ocr_img,
        config=OCR_CONFIG
    )

    texto = re.sub(
        r'[^A-Z0-9]',
        '',
        texto.upper()
    )

    # =========================
    # CLASIFICAR
    # =========================

    ancho = x2 - x1
    largo = y2 - y1

    es_moto = not (ancho > (largo * 2))

    placa_final = validar_placa(
        texto,
        es_moto
    )

    # =========================
    # MOSTRAR RESULTADO
    # =========================

    print("OCR RAW:", texto)

    if placa_final:
        print("PLACA VÁLIDA:", placa_final)
    else:
        print("Placa inválida")

    # =========================
    # DIBUJAR
    # =========================

    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    cv2.putText(
        img,
        texto,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # =========================
    # DEBUG IMAGES
    # =========================

    cv2.imwrite("debug_01_roi.jpg", roi)
    cv2.imwrite("debug_02_mask.jpg", mask)
    cv2.imwrite("debug_03_rectificada.jpg", placa_recta)
    cv2.imwrite("debug_04_ocr.jpg", ocr_img)
    cv2.imwrite("debug_final.jpg", img)

    print("Imágenes debug guardadas")

# ============================================
# START
# ============================================

if __name__ == "__main__":
    main()
