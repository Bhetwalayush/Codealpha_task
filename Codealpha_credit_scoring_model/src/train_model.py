import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import preprocess_data
from feature_engineering import add_features

df = pd.read_csv("data/credit_data.csv")

df = add_features(df)

X_train, X_test, y_train, y_test = preprocess_data(df)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/credit_model.pkl")

print("Model Saved Successfully")