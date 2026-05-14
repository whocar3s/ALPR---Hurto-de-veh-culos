# ====================================
# CÁMARA
# ====================================

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():

    print("No se pudo abrir cámara")
    return

print("Cámara iniciada")

# buffer pequeño para evitar delay
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ====================================
# LOOP
# ====================================

while True:

    ret, img = cap.read()

    if not ret:

        print("Frame perdido")
        continue

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

    # YOLOv8
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
    # SI NO HAY DETECCIONES
    # ====================================

    if len(detecciones) == 0:

        cv2.imshow("ALPR", img)

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

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w_orig, x2)
    y2 = min(h_orig, y2)

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    # ====================================
    # TIPO VEHÍCULO
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

        cv2.imshow("ALPR", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        continue

    # ====================================
    # OCR
    # ====================================

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

        if len(texto) == 6:

            placas_detectadas.append(
                texto
            )

    mejor_valido = None

    if len(placas_detectadas) > 0:

        conteo = {}

        for placa_detectada in placas_detectadas:

            if placa_detectada not in conteo:
                conteo[placa_detectada] = 0

            conteo[placa_detectada] += 1

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

        else:
            mejor_valido = candidata

    # ====================================
    # DIBUJAR
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

    cv2.imshow("ALPR", img)

    # ESC = salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ====================================
# RELEASE
# ====================================

cap.release()
cv2.destroyAllWindows()
