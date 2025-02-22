from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load mô hình đã huấn luyện
model = joblib.load("gold_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        features = [
            float(data["feature1"]),
            float(data["feature2"]),
            float(data["feature3"]),
            float(data["feature4"]),
            float(data["feature5"]),
            float(data["feature6"]),
            float(data["feature7"]),
            float(data["feature8"]),
        ]
        features = np.array(features).reshape(1, -1)
        predicted_price = model.predict(features)[0]
        return render_template("index.html", prediction=predicted_price)
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
