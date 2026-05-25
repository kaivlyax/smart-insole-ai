from flask_login import (

    LoginManager,

    UserMixin,

    login_user,

    login_required,

    logout_user,

    current_user
)

from werkzeug.security import (

    generate_password_hash,

    check_password_hash
)

from flask import redirect

from flask import session
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler

from flask import Flask, render_template, request, jsonify

import joblib
import pandas as pd
import psycopg2

from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# =========================
# FLASK
# =========================

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = 'login'

# =========================
# EMAIL CONFIG
# =========================

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")

EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

ALERT_RECEIVER = "kumar1302.be24@chitkarauniversity.edu.in"

email_sent = False

# =========================
# DATABASE
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS sensor_logs (

    id SERIAL PRIMARY KEY,

    fsr INTEGER,

    prediction TEXT,

    injury_risk BOOLEAN,

    fall_confidence INTEGER,

    stability_score INTEGER,

    movement_intensity INTEGER,

    ax INTEGER,
    ay INTEGER,
    az INTEGER,

    gx INTEGER,
    gy INTEGER,
    gz INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)

""")

conn.commit()

cursor.execute("""

CREATE TABLE IF NOT EXISTS users (

    id SERIAL PRIMARY KEY,

    username TEXT UNIQUE,

    email TEXT UNIQUE,

    password TEXT

)

""")

cursor.execute("""

CREATE TABLE IF NOT EXISTS login_logs (

    id SERIAL PRIMARY KEY,

    username TEXT,

    ip_address TEXT,

    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)

""")

conn.commit()

cursor.close()
conn.close()
# =========================
# BUFFER
# =========================

sensor_buffer = []

# =========================
# MODEL
# =========================

model = joblib.load('model.pkl')

# load_model("lstm_gait_model.h5", compile=False)

# =========================
# GLOBAL VARIABLES
# =========================

latest_fsr_value = 0

pressure_state = "No Pressure"

ml_prediction = "idle"

lstm_prediction = "Normal"

gait_status = "Normal"

injury_risk = False

heavy_pressure_counter = 0

step_count = 0

walking_status = "Idle"

step_detected = False

pressure_history = []
sequence_buffer = []

fall_detected = False

fall_confidence = 0

stability_score = 100

movement_intensity = 0

# MPU VALUES

ax = 0
ay = 0
az = 0

gx = 0
gy = 0
gz = 0

# =========================
# DATABASE WORKER
# =========================

def database_worker():

    global sensor_buffer

    while True:

        try:

            if len(sensor_buffer) > 0:

                local_copy = sensor_buffer.copy()

                sensor_buffer.clear()

                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()

                for item in local_copy:

                    cursor.execute("""

INSERT INTO sensor_logs (

    fsr,
    prediction,
    injury_risk,
    fall_confidence,
    stability_score,
    movement_intensity,

    ax,
    ay,
    az,

    gx,
    gy,
    gz

)

VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)

""", (

    item['fsr'],
    item['prediction'],
    item['injury_risk'],
    item['fall_confidence'],
    item['stability_score'],
    item['movement_intensity'],

    item['ax'],
    item['ay'],
    item['az'],

    item['gx'],
    item['gy'],
    item['gz']

))

                conn.commit()
                cursor.close()
                conn.close()

                print(
                    "Inserted",
                    len(local_copy),
                    "records into Neon"
                )

        except Exception as e:

            print("DB ERROR:", e)

        time.sleep(5)

# =========================
# HOME
# =========================

# =========================
# USER CLASS
# =========================

class User(UserMixin):

    def __init__(self, id, username):

        self.id = id

        self.username = username

# =========================
# USER LOADER
# =========================

@login_manager.user_loader
def load_user(user_id):

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE id=%s",
        (user_id,)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        return User(user[0], user[1])

    return None

@app.route('/')
@login_required
def home():


    return render_template('index.html')

# =========================
# SENSOR API
# =========================

@app.route('/sensor', methods=['POST'])
def sensor():

    global latest_fsr_value
    global pressure_state
    global step_count
    global walking_status
    global step_detected
    global pressure_history
    global fall_detected
    global ml_prediction
    global gait_status
    global injury_risk
    global heavy_pressure_counter
    global movement_intensity
    global fall_confidence
    global stability_score
    global email_sent

    global ax, ay, az
    global gx, gy, gz
    global sequence_buffer
    global lstm_prediction
    data = request.json

    # =========================
    # SENSOR VALUES
    # =========================

    latest_fsr_value = data['fsr']

    ax = data['ax']
    ay = data['ay']
    az = data['az']

    gx = data['gx']
    gy = data['gy']
    gz = data['gz']

    # =========================
    # MOTION ANALYTICS
    # =========================

    accel_magnitude = abs(ax) + abs(ay) + abs(az)

    gyro_magnitude = abs(gx) + abs(gy) + abs(gz)

    movement_intensity = (
        accel_magnitude +
        gyro_magnitude
    )

    current_frame = [

        latest_fsr_value,

        ax,
        ay,
        az,

        gx,
        gy,
        gz,

        movement_intensity,

        stability_score
    ]

    sequence_buffer.append(current_frame)

    if len(sequence_buffer) > 20:

        sequence_buffer.pop(0)

    # =========================
    # STABILITY SCORE
    # =========================

    stability_score = 100

    if movement_intensity > 20000:

        stability_score -= 20

    if gyro_magnitude > 1000:

        stability_score -= 30

    if fall_confidence > 70:

        stability_score -= 40

    if stability_score < 0:

        stability_score = 0

    # =========================
    # PRESSURE HISTORY
    # =========================

    pressure_history.append(
        latest_fsr_value
    )

    if len(pressure_history) > 10:

        pressure_history.pop(0)

    # =========================
    # ML PREDICTION
    # =========================

    input_data = pd.DataFrame(
        [[latest_fsr_value]],
        columns=['fsr']
    )

    prediction = model.predict(input_data)

    ml_prediction = prediction[0]

    print("AI Prediction:", ml_prediction)

# =========================
# LSTM INFERENCE
# =========================

    if  len(sequence_buffer) == 20:

        try:

            sequence_array = np.array(
                sequence_buffer
            )

            scaler = MinMaxScaler()

            scaled_sequence = scaler.fit_transform(
                sequence_array
            )

            X_input = np.expand_dims(
                scaled_sequence,
                axis=0
            )

            prediction = "idle"

            predicted_class = 0

            if predicted_class == 0:

                lstm_prediction = "Normal"

            else:

                lstm_prediction = "Instability"

            print(
                "LSTM Prediction:",
                lstm_prediction
            )

        except Exception as e:
            print("LSTM ERROR:", e)

    # =========================
    # FALL CONFIDENCE ENGINE
    # =========================

    fall_confidence = 0

    if latest_fsr_value > 3000:

        fall_confidence += 30

    if accel_magnitude > 25000:

        fall_confidence += 35

    if gyro_magnitude > 800:

        fall_confidence += 25

    if (
        latest_fsr_value < 200 and
        max(pressure_history) > 3000
    ):

        fall_confidence += 20

    if fall_confidence > 100:

        fall_confidence = 100

    # =========================
    # GAIT STATUS
    # =========================

    if latest_fsr_value > 3500:

        gait_status = "Abnormal"

    else:

        gait_status = "Normal"

    # =========================
    # INJURY RISK
    # =========================

    if latest_fsr_value > 3000:

        heavy_pressure_counter += 1

    else:

        heavy_pressure_counter = 0

    if heavy_pressure_counter >= 5:

        injury_risk = True

        gait_status = "INJURY RISK"

    else:

        injury_risk = False

    # =========================
    # FALL DETECTION
    # =========================

    fall_detected = False

    if fall_confidence >= 70:

        fall_detected = True

        gait_status = "FALL DETECTED"

        injury_risk = True

        print("FALL DETECTED!")

    # =========================
    # EMAIL ALERT
    # =========================

    if fall_detected and not email_sent:

        try:

            subject = "EMERGENCY FALL DETECTED"

            body = f"""
Emergency detected by Smart Insole AI.

Fall Confidence: {fall_confidence}%

FSR Pressure: {latest_fsr_value}

Stability Score: {stability_score}

Movement Intensity: {movement_intensity}

Gait Status: {gait_status}

Immediate attention may be required.
"""

            msg = MIMEMultipart()

            msg['From'] = EMAIL_ADDRESS

            msg['To'] = ALERT_RECEIVER

            msg['Subject'] = subject

            msg.attach(
                MIMEText(
                    body,
                    'plain',
                    'utf-8'
                )
            )

            server = smtplib.SMTP(
                'smtp.gmail.com',
                587
            )

            server.starttls()

            server.login(
                EMAIL_ADDRESS,
                EMAIL_PASSWORD
            )

            server.send_message(msg)

            server.quit()

            print("EMAIL ALERT SENT!")

            email_sent = True

        except Exception as e:

            print("EMAIL ERROR:", e)

    if fall_confidence < 20:

        email_sent = False

    # =========================
    # PRESSURE STATE
    # =========================

    if latest_fsr_value < 150:

        pressure_state = "No Pressure"

    elif latest_fsr_value < 700:

        pressure_state = "Light Pressure"

    else:

        pressure_state = "Heavy Pressure"

    # =========================
    # STEP DETECTION
    # =========================

    STEP_THRESHOLD = 700

    if (
        latest_fsr_value > STEP_THRESHOLD and
        not step_detected
    ):

        step_count += 1

        step_detected = True

        walking_status = "Walking"

    elif latest_fsr_value < 300:

        step_detected = False

        walking_status = "Idle"

    # =========================
    # BUFFER DATABASE INSERT
    # =========================

    sensor_buffer.append({

        "fsr": latest_fsr_value,

        "prediction": ml_prediction,

        "injury_risk": injury_risk,

        "fall_confidence": fall_confidence,

        "stability_score": stability_score,

        "movement_intensity": movement_intensity,



        "ax": ax,
        "ay": ay,
        "az": az,

        "gx": gx,
        "gy": gy,
        "gz": gz
    })

    return jsonify({

        "status": "received",

        "injury_risk": injury_risk

    })

# =========================
# FRONTEND API
# =========================

@app.route('/get_data')
@login_required
def get_data():

    return jsonify({

        'ax': ax,
        'ay': ay,
        'az': az,

        'gx': gx,
        'gy': gy,
        'gz': gz,

        'fsr': latest_fsr_value,

        'state': pressure_state,

        'steps': step_count,

        'walking': walking_status,

        'prediction': ml_prediction,

        'gait': gait_status,

        'injury_risk': injury_risk,

        'fall': fall_detected,

        'fall_confidence': fall_confidence,

        'stability_score': stability_score,

        'movement_intensity': movement_intensity,
        
        'lstm_prediction': lstm_prediction,

    })

@app.route('/register', methods=['GET','POST'])
def register():

    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']

        password = generate_password_hash(
            request.form['password']
        )

        try:

            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()

            cursor.execute(
                """
INSERT INTO users (
    username,
    email,
    password
)
VALUES (%s,%s,%s)
                """,
                (username, email, password)
            )

            conn.commit()

            cursor.close()
            conn.close()

            return redirect('/login')

        except Exception as e:

            print("REGISTER ERROR:", e)

            return str(e)

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])

def login():

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute(

            "SELECT * FROM users WHERE username=%s",

            (username,)
        )

        user = cursor.fetchone()

        if user and check_password_hash(

            user[3],
            password
        ):

            login_user(

                User(user[0], user[1])
            )

            # LOGIN LOG

            cursor.execute(

                """

INSERT INTO login_logs (

    username,
    ip_address

)

VALUES (%s,%s)

                """,

                (
                    username,
                    request.remote_addr
                )
            )

            conn.commit()

            cursor.close()
            conn.close()
           

            return redirect('/')
        
            
        cursor.close()
        conn.close()
    return render_template('login.html')

@app.route('/logout')

@login_required
def logout():

    logout_user()

    return redirect('/login')

# =========================
# START SERVER
# =========================
threading.Thread(target=database_worker, daemon=True).start()
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
