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
        # Nếu request từ form HTML
        if request.form:
            features = [
                float(request.form["feature1"]),
                float(request.form["feature2"]),
                float(request.form["feature3"]),
                float(request.form["feature4"]),
                float(request.form["feature5"]),
                float(request.form["feature6"]),
                float(request.form["feature7"]),
                float(request.form["feature8"])
            ]
            predicted_price = model.predict(np.array(features).reshape(1, -1))[0]
            return render_template("index.html", predicted_price=predicted_price)

        # Nếu request từ API (JSON)
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input format. Expecting JSON with key 'features'"}), 400
        
        features = np.array(data["features"]).reshape(1, -1)
        predicted_price = model.predict(features)[0]

        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
