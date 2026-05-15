import cv2
import time
import statistics
import subprocess

# ============================================
# CONFIG
# ============================================

IMAGE_PATH = "placa.jpg"

ITERACIONES = 30

# ============================================
# MÉTRICAS
# ============================================

tiempos_captura = []
tiempos_yolo = []
tiempos_ocr = []
tiempos_total = []

cpu_usos = []
ram_usos = []
temperaturas = []

# ============================================
# LEER IMAGEN UNA VEZ
# ============================================

img = cv2.imread(IMAGE_PATH)

if img is None:

    print("No se pudo cargar imagen")
    exit()

print("\n==============================")
print("INICIANDO PRUEBA 1 IMAGEN x30")
print("==============================\n")

# ============================================
# LOOP
# ============================================

for i in range(ITERACIONES):

    inicio_total = time.time()

    # ====================================
    # CAPTURA
    # ====================================

    inicio_captura = time.time()

    frame = img.copy()

    fin_captura = time.time()

    tiempo_captura = (
        fin_captura - inicio_captura
    )

    tiempos_captura.append(
        tiempo_captura
    )

    # ====================================
    # YOLO
    # ====================================

    inicio_yolo = time.time()

    # ====================================
    # AQUÍ LLAMAS TU FUNCIÓN YOLO
    # ====================================

    # ejemplo:
    #
    # roi = detectar_placa(frame)
    #
    # o simplemente:
    #
    time.sleep(0.15)

    fin_yolo = time.time()

    tiempo_yolo = (
        fin_yolo - inicio_yolo
    )

    tiempos_yolo.append(
        tiempo_yolo
    )

    # ====================================
    # OCR
    # ====================================

    inicio_ocr = time.time()

    # ====================================
    # AQUÍ TU OCR
    # ====================================

    # ejemplo:
    #
    # placa = leer_placa(roi)
    #
    time.sleep(0.08)

    placa = "ABC123"

    fin_ocr = time.time()

    tiempo_ocr = (
        fin_ocr - inicio_ocr
    )

    tiempos_ocr.append(
        tiempo_ocr
    )

    # ====================================
    # TOTAL
    # ====================================

    fin_total = time.time()

    tiempo_total = (
        fin_total - inicio_total
    )

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

        usados = int(line.split()[2])

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

    # ====================================
    # PRINT
    # ====================================

    print(f"\nIteración {i+1}")

    print(
        f"Placa: {placa}"
    )

    print(
        f"Captura: {tiempo_captura:.3f}s"
    )

    print(
        f"YOLO: {tiempo_yolo:.3f}s"
    )

    print(
        f"OCR: {tiempo_ocr:.3f}s"
    )

    print(
        f"Total: {tiempo_total:.3f}s"
    )

    print(
        f"CPU: {cpu_use:.1f}%"
    )

    print(
        f"RAM: {usados} MB"
    )

    print(
        f"Temp: {temp:.1f}°C"
    )

# ============================================
# RESULTADOS FINALES
# ============================================

print("\n===================================")
print("RESULTADOS FINALES")
print("===================================\n")

print(
    f"Captura promedio: "
    f"{statistics.mean(tiempos_captura):.3f}s"
)

print(
    f"YOLO promedio: "
    f"{statistics.mean(tiempos_yolo):.3f}s"
)

print(
    f"OCR promedio: "
    f"{statistics.mean(tiempos_ocr):.3f}s"
)

print(
    f"Tiempo total promedio: "
    f"{statistics.mean(tiempos_total):.3f}s"
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
    f"Temperatura promedio: "
    f"{statistics.mean(temperaturas):.2f} °C"
)

print("\n===================================\n")
