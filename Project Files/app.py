# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["age"]),
                int(request.form["gender"]),
                float(request.form["alcohol_years"]),
                float(request.form["alcohol_quantity"]),
                int(request.form["diabetes"]),
                float(request.form["bp"]),
                int(request.form["obesity"]),
                float(request.form["hb"]),
                float(request.form["pcv"]),
                float(request.form["sgot"]),
                float(request.form["sgpt"]),
                float(request.form["albumin"]),
            ]
            scaled_input = scaler.transform([features])
            proba = model.predict_proba(scaled_input)[0][1]  # probability of cirrhosis
            probability = round(proba * 100, 2)
            prediction = ("🟥 Patient has Liver Cirrhosis"
                          if proba > 0.6
                          else "✅ Patient does NOT have Liver Cirrhosis")
        except Exception as e:
            prediction = f"⚠️ Error in input: {e}"
    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
