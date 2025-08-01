from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model/lstm_multiplier_model.h5")
scaler = joblib.load("model/lstm_multiplier_scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("history")
        if not data or len(data) < 5:
            return jsonify({"error": "Need at least 5 recent multipliers"}), 400
        sequence = np.array(data[-5:]).reshape(-1, 1)
        sequence_scaled = scaler.transform(sequence).reshape(1, 5, 1)
        prediction_scaled = model.predict(sequence_scaled)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
        return jsonify({"prediction": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
