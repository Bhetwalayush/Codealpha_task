from flask import Flask
from flask import request
from flask import jsonify

import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load(
    "models/credit_model.pkl"
)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    income = data["income"]
    age = data["age"]
    loan_amount = data["loan_amount"]
    debt = data["debt"]
    credit_history = data["credit_history"]
    missed_payments = data["missed_payments"]

    debt_income = debt / income
    loan_income = loan_amount / income

    features = np.array([
        [
            income,
            age,
            loan_amount,
            debt,
            credit_history,
            missed_payments,
            debt_income,
            loan_income
        ]
    ])

    result = model.predict(features)[0]

    return jsonify({
        "creditworthiness": int(result)
    })

if __name__ == "__main__":
    app.run(debug=True)