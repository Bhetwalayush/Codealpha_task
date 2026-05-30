import joblib
import numpy as np

model = joblib.load(
    "models/credit_model.pkl"
)

sample = np.array([
    [
        50000,
        30,
        10000,
        4000,
        7,
        0,
        0.08,
        0.20
    ]
])

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Good Credit Risk")
else:
    print("Bad Credit Risk")