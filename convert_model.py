from tensorflow.keras.models import load_model

model = load_model("lstm_gait_model.keras", compile=False)

model.save("lstm_gait_model.h5")

print("Model converted successfully!")