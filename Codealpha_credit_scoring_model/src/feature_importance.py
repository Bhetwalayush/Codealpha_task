import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import preprocess_data
from feature_engineering import add_features

df = pd.read_csv("data/credit_data.csv")

df = add_features(df)

X = df.drop("credit_score", axis=1)
y = df["credit_score"]

X_train, X_test, y_train, y_test = preprocess_data(df)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importance)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Credit Scoring Feature Importance")

plt.tight_layout()
plt.show()