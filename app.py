import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Cargar el modelo
model = tf.keras.models.load_model("housing_price_model_v1.keras")

# Valores usados para normalizar el precio
precio_mean = 4766729.247706422
precio_std = 1870439.6156573922

# Crear la aplicación Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener los datos del request
        data = request.get_json()

        # Convertir a numpy array con tipo float64 para evitar problemas con la serialización
        features = np.array(data["features"], dtype=np.float64).reshape(1, -1)

        # Hacer la predicción
        prediction = model.predict(features)

        # Desnormalizar la predicción
        real_price = (prediction[0, 0] * precio_std) + precio_mean

        # Devolver el resultado en formato JSON con la predicción normalizada y el precio real
        return jsonify({
            "normalized_price": float(prediction[0, 0]),
            "real_price": float(real_price)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
