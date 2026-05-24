import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import joblib

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv('data/sensor_data.csv')

# =========================
# INPUTS AND LABELS
# =========================

X = data[['fsr']]

y = data['label']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = RandomForestClassifier()

model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================

accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, 'model.pkl')

print("Model saved as model.pkl")