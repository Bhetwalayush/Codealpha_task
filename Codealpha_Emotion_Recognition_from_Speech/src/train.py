import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical

from data_loader import load_dataset
from feature_extraction import extract_features
from model import build_model

print("Loading dataset...")

df = load_dataset("dataset/RAVDESS")

X = []
y = []

for index, row in df.iterrows():

    features = extract_features(
        row["path"]
    )

    X.append(features)
    y.append(row["emotion"])

X = np.array(X)

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "saved_model/label_encoder.pkl"
)

y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42
)

model = build_model(
    input_shape=X.shape[1],
    num_classes=y_cat.shape[1]
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Accuracy:", accuracy)

model.save(
    "saved_model/emotion_model.h5"
)