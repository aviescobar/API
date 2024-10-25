
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image


app = Flask(__name__)
# Función para procesar la imagen
def process_image(image_data):
    # Leer la imagen desde los datos recibidos
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)

   # Redimensionar la imagen a (96, 96)
    resized_image = cv2.resize(image, (96, 96))

   # Convertir la imagen redimensionada a escala de grises
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

     # Inicializar el detector ORB
    orb = cv2.ORB_create()

     # Detectar puntos clave y descriptores
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    # Dibujar "X" rojas en las posiciones de los puntos clave

    plt.figure(figsize=(7,7))
    plt.imshow(gray_image, cmap='gray')  # Mostrar la imagen en escala de grises

     for kp in keypoints:
        x, y = kp.pt
        plt.plot(x, y, 'rx')  # 'rx' indica color rojo y marca tipo "X"

     plt.axis('off')  # No mostrar ejes

    # Convertir el gráfico de matplotlib en imagen y luego a base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64

# Ruta para la página HTML
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar la imagen
@app.route('/procesar_imagen', methods=['POST'])
def procesar_imagen():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
     file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Leer la imagen en binario
    image_data = file.read()

      # Procesar la imagen y obtener el resultado
    processed_image_base64 = process_image(image_data)

     # Devolver la imagen procesada en base64 para mostrarla en el frontend
    return jsonify({'image': processed_image_base64})

    if __name__ == '__main__':
        app.run(debug=True)





