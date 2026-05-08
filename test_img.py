import cv2
import numpy as np
import pytesseract
import tflite_runtime.interpreter as tflite

# =========================================================
# CONFIGURACIÓN
# =========================================================

MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "tu_foto_de_placa.jpg"

CONF_THRESHOLD = 0.01

# OCR
OCR_CONFIG = (
    "--psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# =========================================================
# FUNCIONES
# =========================================================

def detectar_esquinas_amarillas(img):

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # rango amarillo
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([40, 255, 255])

    # máscara
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # limpiar
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

    # coordenadas blancas
    ys, xs = np.where(mask == 255)

    # si no encuentra nada
    if len(xs) == 0:
        return None

    points = np.column_stack((xs, ys))

    # esquinas
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


def corregir_perspectiva(img, corners):

    width = 300
    height = 100

    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    M = cv2.getPerspectiveTransform(corners, pts2)

    dst = cv2.warpPerspective(
        img,
        M,
        (width, height)
    )

    return dst


def preprocess_ocr(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2
    )

    # blur
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # threshold adaptativo
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh


def validar_formato(texto, es_carro):

    texto = texto.replace(" ", "").strip()

    if len(texto) != 6:
        return False

    # 3 primeras letras
    if not texto[:3].isalpha():
        return False

    # siguientes 2 números
    if not texto[3:5].isdigit():
        return False

    ultimo = texto[5]

    # carro
    if es_carro:
        return ultimo.isdigit()

    # moto
    return ultimo in "ABCDEFGHI"


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # CARGAR MODELO
    # =====================================================

    print(f"Cargando modelo: {MODEL_PATH}")

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, input_h, input_w, _ = input_details[0]['shape']

    # =====================================================
    # CARGAR IMAGEN
    # =====================================================

    original_img = cv2.imread(IMAGE_PATH)

    if original_img is None:
        print("Error cargando imagen")
        return

    h_orig, w_orig = original_img.shape[:2]

    # =====================================================
    # PREPROCESS YOLO
    # =====================================================

    rgb_img = cv2.cvtColor(
        original_img,
        cv2.COLOR_BGR2RGB
    )

    input_img = cv2.resize(
        rgb_img,
        (input_w, input_h)
    )

    input_img = input_img.astype(np.float32) / 255.0

    input_img = np.expand_dims(input_img, axis=0)

    # =====================================================
    # INFERENCIA
    # =====================================================

    interpreter.set_tensor(
        input_details[0]['index'],
        input_img
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]['index']
    )

    # =====================================================
    # POSTPROCESS
    # =====================================================

    output = np.squeeze(output)

    if output.shape[0] == 5:
        output = output.T

    # mejor detección
    best_detection = max(output, key=lambda x: x[4])

    conf = best_detection[4]

    print(f"Mejor confianza: {conf:.4f}")

    if conf < CONF_THRESHOLD:
        print("No se detectó placa")
        return

    cx, cy, w, h = best_detection[:4]

    # coordenadas
    if np.max(best_detection[:4]) <= 1.1:

        x1 = int((cx - w/2) * w_orig)
        y1 = int((cy - h/2) * h_orig)
        x2 = int((cx + w/2) * w_orig)
        y2 = int((cy + h/2) * h_orig)

    else:

        x1 = int((cx - w/2) * (w_orig / input_w))
        y1 = int((cy - h/2) * (h_orig / input_h))
        x2 = int((cx + w/2) * (w_orig / input_w))
        y2 = int((cy + h/2) * (h_orig / input_h))

    # límites
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_orig, x2)
    y2 = min(h_orig, y2)

    print(f"Placa: {x1}, {y1}, {x2}, {y2}")

    # =====================================================
    # RECORTE
    # =====================================================

    placa = original_img[y1:y2, x1:x2]

    # =====================================================
    # DETECTAR ESQUINAS
    # =====================================================

    corners = detectar_esquinas_amarillas(placa)

    if corners is None:
        print("No se encontraron esquinas")
        return

    # =====================================================
    # CORREGIR PERSPECTIVA
    # =====================================================

    placa_recta = corregir_perspectiva(
        placa,
        corners
    )

    # =====================================================
    # OCR
    # =====================================================

    pre = preprocess_ocr(placa_recta)

    texto = pytesseract.image_to_string(
        pre,
        config=OCR_CONFIG
    )

    texto = texto.replace("\n", "").replace(" ", "")

    print("OCR:", texto)

    # =====================================================
    # DETERMINAR CARRO / MOTO
    # =====================================================

    ancho = x2 - x1
    alto = y2 - y1

    es_carro = ancho > (alto * 2)

    print("Tipo:", "CARRO" if es_carro else "MOTO")

    valido = validar_formato(
        texto,
        es_carro
    )

    print("Formato válido:", valido)

    # =====================================================
    # DIBUJAR
    # =====================================================

    cv2.rectangle(
        original_img,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        3
    )

    cv2.putText(
        original_img,
        texto,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # =====================================================
    # GUARDAR
    # =====================================================

    cv2.imwrite("resultado_final.jpg", original_img)

    cv2.imwrite("placa_recta.jpg", placa_recta)

    cv2.imwrite("ocr_preprocess.jpg", pre)

    print("Resultado guardado")


if __name__ == "__main__":
    main()
