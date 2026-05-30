import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression

from data_preprocessing import preprocess_data
from feature_engineering import add_features

df = pd.read_csv("data/credit_data.csv")

df = add_features(df)

X_train, X_test, y_train, y_test = preprocess_data(df)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:,1]

print(
    "ROC AUC:",
    roc_auc_score(y_test, y_prob)
)

print(
    "Confusion Matrix:\n",
    confusion_matrix(y_test, y_pred)
)