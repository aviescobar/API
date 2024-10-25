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


