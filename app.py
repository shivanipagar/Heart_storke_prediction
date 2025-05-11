from flask import Flask, render_template, request, jsonify
from model import predict_stroke, recommend_hospitals

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/input", methods=["GET"])
def input_page():
    return render_template("input.html")

@app.route("/prediction", methods=["GET"])
def prediction_page():
    return render_template("prediction.html")

@app.route("/page4", methods=["GET"])
def page4():
    return render_template("page4.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect data from JSON (sent by JS in input.html)
        data = request.get_json()

        # Convert values to match expected types
        user_input = {
            'gender': data['gender'],
            'age': float(data['age']),
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease']),
            'ever_married': data['ever_married'],
            'work_type': data['work_type'],
            'Residence_type': data['residence_type'],
            'avg_glucose_level': float(data['avg_glucose_level']),
            'bmi': float(data['bmi']),
            'smoking_status': data['smoking_status']
        }

        probability, prediction = predict_stroke(user_input)
        # risk_level = "High" if prediction == 1 else "Low"
        risk_level = "High" if probability >= 0.3 else "Low"

        

        return jsonify({
            "probability": probability,
            "prediction": int(prediction),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_hospitals')
def get_hospitals():
    city = request.args.get('city', '')
    local_address = request.args.get('localAddress', '')
    hospitals = recommend_hospitals(city, local_address)
    return jsonify({'hospitals': hospitals})

if __name__ == "__main__":
    app.run(debug=True)
