import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Silenciar advertencias de TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Cargar el modelo
model = tf.keras.models.load_model("housing_price_model_v1.keras")

# Valores para normalización
precio_mean = 4766729.247706422
precio_std = 1870439.6156573922

# Error estándar estimado (ajusta este valor según tu modelo)
error_std = 0.15  # 15% de error estándar estimado

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validar datos de entrada
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Se requieren 'features' en el JSON"}), 400
            
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)
        
        # Verificar dimensiones
        if features.shape[1] != model.input_shape[1]:
            return jsonify({
                "error": f"Dimensión incorrecta. Esperado {model.input_shape[1]} características, recibido {features.shape[1]}"
            }), 400

        # Predicción principal
        prediction = model.predict(features, verbose=0)[0][0]
        
        # Calcular intervalo de confianza (95%)
        lower_bound = prediction * (1 - 1.96 * error_std)
        upper_bound = prediction * (1 + 1.96 * error_std)
        
        # Desnormalizar valores
        real_price = (prediction * precio_std) + precio_mean
        real_lower = (lower_bound * precio_std) + precio_mean
        real_upper = (upper_bound * precio_std) + precio_mean

        return jsonify({
            "normalized_price": float(prediction),
            "price_range_normalized": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            },
            "real_price": float(real_price),
            "real_price_range": {
                "lower": float(real_lower),
                "upper": float(real_upper)
            },
            "confidence_level": "95%",
            "model_error_std": float(error_std)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "stack_trace": str(e.__traceback__)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)