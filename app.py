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


