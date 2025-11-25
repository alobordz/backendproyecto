import cv2
import mediapipe as mp
import numpy as np
import os
from flask_cors import CORS


from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

# -----------------------------
# Configuración básica Flask
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


UPLOAD_FOLDER = "uploads_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# URL base del backend:
# - En local:   http://127.0.0.1:5000
# - En Render:  https://TU-NOMBRE-APP.onrender.com  (en el futuro)
BASE_URL = "http://127.0.0.1:5000"

# -----------------------------
# Inicializar MediaPipe FaceMesh
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------------------
# Funciones auxiliares
# -----------------------------
def distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def ojo_cerrado(landmarks, width, height):
    # Índices de párpados según FaceMesh
    ojo_derecho = [159, 145]
    ojo_izquierdo = [386, 374]

    parpado_derecho = distancia(
        (landmarks.landmark[ojo_derecho[0]].x * width,
         landmarks.landmark[ojo_derecho[0]].y * height),
        (landmarks.landmark[ojo_derecho[1]].x * width,
         landmarks.landmark[ojo_derecho[1]].y * height)
    )

    parpado_izquierdo = distancia(
        (landmarks.landmark[ojo_izquierdo[0]].x * width,
         landmarks.landmark[ojo_izquierdo[0]].y * height),
        (landmarks.landmark[ojo_izquierdo[1]].x * width,
         landmarks.landmark[ojo_izquierdo[1]].y * height)
    )

    # Umbral simple para ojo casi cerrado
    return (parpado_derecho < 4 and parpado_izquierdo < 4)

def detectar_direccion(landmarks, width, height):
    # Ojos (esquinas) + iris
    ojo_derecho = [33, 133]
    ojo_izquierdo = [362, 263]
    iris_derecho = 468
    iris_izquierdo = 473

    puntos = {}
    for i in ojo_derecho + ojo_izquierdo + [iris_derecho, iris_izquierdo]:
        x = int(landmarks.landmark[i].x * width)
        y = int(landmarks.landmark[i].y * height)
        puntos[i] = (x, y)

    # Centro del ojo izquierdo (promedio de esquinas 362 y 263)
    x_centro_ojo = np.mean([puntos[362][0], puntos[263][0]])
    y_centro_ojo = np.mean([puntos[362][1], puntos[263][1]])

    dx = puntos[iris_izquierdo][0] - x_centro_ojo
    dy = puntos[iris_izquierdo][1] - y_centro_ojo

    if dx > 5:
        return "Si", dx, dy
    elif dx < -5:
        return "no", dx, dy
    elif dy < -3:
        return "Ayuda", dx, dy
    else:
        return "Centro", dx, dy

def analizar_imagen(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return {
            "direction": "error",
            "message": "No se pudo leer la imagen."
        }

    height, width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return {
            "direction": "error",
            "message": "No se detectó ningún rostro en la imagen."
        }

    face_landmarks = results.multi_face_landmarks[0]

    # Revisar cierre de ojos -> "gracias"
    if ojo_cerrado(face_landmarks, width, height):
        direction = "gracias"
    else:
        direction, dx, dy = detectar_direccion(face_landmarks, width, height)

    mensajes = {
        "Si": "Se detectó que estás mirando hacia la DERECHA (comando: SI).",
        "no": "Se detectó que estás mirando hacia la IZQUIERDA (comando: NO).",
        "Ayuda": "Se detectó que estás mirando hacia ARRIBA (comando: AYUDA).",
        "Centro": "Se detectó que tu mirada está CENTRADA (comando: CENTRO).",
        "gracias": "Se detectó que tus ojos están CERRADOS (comando: GRACIAS)."
    }

    message = mensajes.get(direction, f"Dirección detectada: {direction}")

    return {
        "direction": direction,
        "message": message
    }

# -----------------------------
# Rutas Flask
# -----------------------------
HOME_HTML = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <title>Prueba local - Detección de mirada</title>
  </head>
  <body>
    <h1>Prueba local - Backend detección de mirada</h1>
    <p>Sube una foto donde se vea tu cara y tu mirada (derecha, izquierda, arriba, centro o ojos cerrados).</p>
    <form action="/process_image" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <button type="submit">Enviar imagen</button>
    </form>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    # Paginita simple para probar rápido en el navegador
    return render_template_string(HOME_HTML)

@app.route("/process_image", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    resultado = analizar_imagen(filepath)

    status_code = 200 if resultado.get("direction") != "error" else 400
    return jsonify(resultado), status_code

if __name__ == "__main__":
    # En local:
    app.run(host="127.0.0.1", port=5000, debug=True)