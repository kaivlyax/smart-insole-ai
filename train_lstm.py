import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("sensor_logs.csv")

print(df.head())

# =========================
# CREATE LABELS
# =========================

labels = []

for confidence in df['fall_confidence']:

    if confidence >= 70:

        labels.append(2)

    elif confidence >= 30:

        labels.append(1)

    else:

        labels.append(0)

df['label'] = labels

# =========================
# FEATURES
# =========================

features = [

    'fsr',

    'ax',
    'ay',
    'az',

    'gx',
    'gy',
    'gz',

    'movement_intensity',

    'stability_score'
]

# =========================
# NORMALIZE
# =========================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    df[features]
)

# =========================
# CREATE SEQUENCES
# =========================

SEQUENCE_LENGTH = 20

X = []
y = []

for i in range(
    len(scaled_data) - SEQUENCE_LENGTH
):

    X.append(
        scaled_data[
            i:i + SEQUENCE_LENGTH
        ]
    )

    y.append(
        df['label'].iloc[
            i + SEQUENCE_LENGTH
        ]
    )

X = np.array(X)

y = np.array(y)

# =========================
# ONE HOT ENCODE
# =========================

y = to_categorical(y)

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
# BUILD LSTM MODEL
# =========================

model = Sequential()

model.add(

    LSTM(

        64,

        return_sequences=True,

        input_shape=(
            X.shape[1],
            X.shape[2]
        )
    )
)

model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax'))

# =========================
# COMPILE
# =========================

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================

history = model.fit(

    X_train,
    y_train,

    epochs=20,

    batch_size=32,

    validation_data=(X_test, y_test)
)

# =========================
# EVALUATE
# =========================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("LSTM Accuracy:", accuracy)

# =========================
# SAVE MODEL
# =========================

model.save("lstm_gait_model.keras")

print("LSTM model saved!")