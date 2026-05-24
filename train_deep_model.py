import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA
# =========================

data = pd.read_csv('data/sensor_data.csv')

# =========================
# INPUT FEATURES
# =========================

X = data[['fsr']].values

# =========================
# LABELS
# =========================

y = data['label']

# =========================
# ENCODE LABELS
# =========================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

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
# BUILD DEEP MODEL
# =========================

model = Sequential()

model.add(Dense(
    32,
    activation='relu',
    input_shape=(1,)
))

model.add(Dense(
    16,
    activation='relu'
))

model.add(Dense(
    3,
    activation='softmax'
))

# =========================
# COMPILE
# =========================

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================

model.fit(

    X_train,
    y_train,

    epochs=20
)

# =========================
# EVALUATE
# =========================

loss, accuracy = model.evaluate(

    X_test,
    y_test
)

print("Deep Learning Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================

model.save('deep_model.h5')

print("Deep model saved")