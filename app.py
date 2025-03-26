import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Cargar el modelo
model = tf.keras.models.load_model("modelo.keras")

# Crear la aplicación Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener los datos del request
        data = request.get_json()
        
        # Convertir a numpy array
        features = np.array(data["features"]).reshape(1, -1)
        
        # Hacer la predicción
        prediction = model.predict(features)
        
        # Devolver el resultado en formato JSON
        return jsonify({"prediction": float(prediction[0, 0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
