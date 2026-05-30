from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask import redirect, session, flash, url_for
from authlib.integrations.flask_client import OAuth
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
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")

# =========================
# GOOGLE OAUTH
# =========================
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# =========================
# LOGIN MANAGER
# =========================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

latest_data = {
    "fsr": 0, "ax": 0, "ay": 0, "az": 0,
    "gx": 0, "gy": 0, "gz": 0,
    "fall": False, "fall_confidence": 0,
    "movement_intensity": 0,
    "lstm_prediction": "Normal", "injury_risk": False
}

# =========================
# EMAIL CONFIG
# =========================
EMAIL_ADDRESS  = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ALERT_RECEIVER = "kumar1302.be24@chitkarauniversity.edu.in"
email_sent = False

# =========================
# DATABASE SETUP
# =========================
def get_db():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn   = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_logs (
            id SERIAL PRIMARY KEY,
            fsr INTEGER, prediction TEXT, injury_risk BOOLEAN,
            fall_confidence INTEGER, stability_score INTEGER,
            movement_intensity INTEGER,
            ax INTEGER, ay INTEGER, az INTEGER,
            gx INTEGER, gy INTEGER, gz INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            full_name TEXT,
            phone TEXT,
            google_id TEXT UNIQUE,
            avatar_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrate: add new columns if they don't exist
    for col, coltype in [
        ("full_name", "TEXT"), ("phone", "TEXT"),
        ("google_id", "TEXT"), ("avatar_url", "TEXT"),
        ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    ]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {coltype}")
        except Exception:
            conn.rollback()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id SERIAL PRIMARY KEY,
            username TEXT, ip_address TEXT,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()

init_db()

# =========================
# BUFFER & GLOBALS
# =========================
sensor_buffer        = []
model                = joblib.load('model.pkl')
latest_fsr_value     = 0
pressure_state       = "No Pressure"
ml_prediction        = "idle"
lstm_prediction      = "Normal"
gait_status          = "Normal"
injury_risk          = False
heavy_pressure_counter = 0
step_count           = 0
walking_status       = "Idle"
step_detected        = False
pressure_history     = []
sequence_buffer      = []
fall_detected        = False
fall_confidence      = 0
stability_score      = 100
movement_intensity   = 0
ax = ay = az = gx = gy = gz = 0

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
                conn   = get_db()
                cursor = conn.cursor()
                for item in local_copy:
                    cursor.execute("""
                        INSERT INTO sensor_logs (
                            fsr, prediction, injury_risk, fall_confidence,
                            stability_score, movement_intensity,
                            ax, ay, az, gx, gy, gz
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        item['fsr'], item['prediction'], item['injury_risk'],
                        item['fall_confidence'], item['stability_score'],
                        item['movement_intensity'],
                        item['ax'], item['ay'], item['az'],
                        item['gx'], item['gy'], item['gz']
                    ))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Inserted {len(local_copy)} records")
        except Exception as e:
            print("DB ERROR:", e)
        time.sleep(5)

# =========================
# USER CLASS
# =========================
class User(UserMixin):
    def __init__(self, id, username, email=None, full_name=None,
                 phone=None, avatar_url=None, created_at=None):
        self.id         = id
        self.username   = username
        self.email      = email
        self.full_name  = full_name
        self.phone      = phone
        self.avatar_url = avatar_url
        self.created_at = created_at

@login_manager.user_loader
def load_user(user_id):
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    u = cursor.fetchone()
    cursor.close(); conn.close()
    if u:
        return User(u[0], u[1], u[2],
                    u[4] if len(u) > 4 else None,
                    u[5] if len(u) > 5 else None,
                    u[7] if len(u) > 7 else None,
                    u[8] if len(u) > 8 else None)
    return None

# =========================
# ROUTES
# =========================

@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user)

# ── Login ──
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')

        conn   = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()

        if not user:
            flash('No account found with that username.', 'error')
            cursor.close(); conn.close()
            return render_template('login.html')

        if not check_password_hash(user[3], password):
            flash('Incorrect password. Please try again.', 'error')
            cursor.close(); conn.close()
            return render_template('login.html')

        login_user(User(user[0], user[1], user[2],
                        user[4] if len(user) > 4 else None,
                        user[5] if len(user) > 5 else None,
                        user[7] if len(user) > 7 else None,
                        user[8] if len(user) > 8 else None))

        cursor.execute(
            "INSERT INTO login_logs (username, ip_address) VALUES (%s,%s)",
            (username, request.remote_addr)
        )
        conn.commit()
        cursor.close(); conn.close()
        return redirect('/')

    return render_template('login.html')

# ── Register ──
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email    = request.form['email'].strip()
        password = request.form['password']
        confirm  = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('register.html')

        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template('register.html')

        if confirm and password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn   = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            flash('Username already taken. Please choose another.', 'error')
            cursor.close(); conn.close()
            return render_template('register.html')

        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash('An account with that email already exists.', 'error')
            cursor.close(); conn.close()
            return render_template('register.html')

        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s,%s,%s)",
                (username, email, generate_password_hash(password))
            )
            conn.commit()
            flash('Account created! Please sign in.', 'success')
            cursor.close(); conn.close()
            return redirect('/login')
        except Exception as e:
            print("REGISTER ERROR:", e)
            flash('Registration failed. Please try again.', 'error')
            cursor.close(); conn.close()
            return render_template('register.html')

    return render_template('register.html')

# ── Google OAuth ──
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    try:
        token    = google.authorize_access_token()
        userinfo = token.get('userinfo')
        if not userinfo:
            flash('Google login failed. Try again.', 'error')
            return redirect('/login')

        google_id  = userinfo['sub']
        email      = userinfo.get('email', '')
        name       = userinfo.get('name', '')
        avatar_url = userinfo.get('picture', '')
        username   = email.split('@')[0].replace('.', '_')

        conn   = get_db()
        cursor = conn.cursor()

        # Check if google_id already registered
        cursor.execute("SELECT * FROM users WHERE google_id=%s", (google_id,))
        user = cursor.fetchone()

        if not user:
            # Check by email
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
            if user:
                # Link google_id to existing account
                cursor.execute(
                    "UPDATE users SET google_id=%s, avatar_url=%s WHERE id=%s",
                    (google_id, avatar_url, user[0])
                )
                conn.commit()
            else:
                # New user via Google
                base_username = username
                counter = 1
                while True:
                    cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
                    if not cursor.fetchone():
                        break
                    username = f"{base_username}{counter}"
                    counter += 1

                cursor.execute("""
                    INSERT INTO users (username, email, password, full_name, google_id, avatar_url)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (username, email, generate_password_hash(os.urandom(24).hex()),
                      name, google_id, avatar_url))
                conn.commit()
                cursor.execute("SELECT * FROM users WHERE google_id=%s", (google_id,))
                user = cursor.fetchone()

        cursor.execute("SELECT * FROM users WHERE google_id=%s", (google_id,))
        user = cursor.fetchone()

        login_user(User(user[0], user[1], user[2],
                        user[4] if len(user) > 4 else None,
                        user[5] if len(user) > 5 else None,
                        user[7] if len(user) > 7 else None,
                        user[8] if len(user) > 8 else None))

        cursor.execute(
            "INSERT INTO login_logs (username, ip_address) VALUES (%s,%s)",
            (user[1], request.remote_addr)
        )
        conn.commit()
        cursor.close(); conn.close()
        return redirect('/')

    except Exception as e:
        print("GOOGLE AUTH ERROR:", e)
        flash('Google login failed. Please try again.', 'error')
        return redirect('/login')

# ── Logout ──
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

# ── Profile ──
@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        phone     = request.form.get('phone', '').strip()
        email     = request.form.get('email', '').strip()

        conn   = get_db()
        cursor = conn.cursor()

        # Check email not taken by another user
        if email:
            cursor.execute("SELECT id FROM users WHERE email=%s AND id!=%s", (email, current_user.id))
            if cursor.fetchone():
                flash('That email is already in use.', 'error')
                cursor.close(); conn.close()
                return redirect('/profile')

        cursor.execute("""
            UPDATE users SET full_name=%s, phone=%s, email=%s WHERE id=%s
        """, (full_name, phone, email, current_user.id))
        conn.commit()
        cursor.close(); conn.close()
        flash('Profile updated successfully.', 'success')
        return redirect('/profile')

    # Get fresh user data
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=%s", (current_user.id,))
    u = cursor.fetchone()

    # Get login history
    cursor.execute("""
        SELECT ip_address, login_time FROM login_logs
        WHERE username=%s ORDER BY login_time DESC LIMIT 5
    """, (current_user.username,))
    login_history = cursor.fetchall()

    # Get session count
    cursor.execute("SELECT COUNT(*) FROM login_logs WHERE username=%s", (current_user.username,))
    session_count = cursor.fetchone()[0]

    cursor.close(); conn.close()

    user_data = {
        'id': u[0], 'username': u[1], 'email': u[2],
        'full_name': u[4] if len(u) > 4 else '',
        'phone': u[5] if len(u) > 5 else '',
        'avatar_url': u[7] if len(u) > 7 else '',
        'created_at': u[8] if len(u) > 8 else ''
    }
    return render_template('profile.html', user=user_data,
                           login_history=login_history,
                           session_count=session_count)

# ── History ──
@app.route('/history')
@login_required
def history():
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, fsr, prediction, injury_risk, fall_confidence,
               stability_score, movement_intensity, created_at
        FROM sensor_logs ORDER BY created_at DESC LIMIT 100
    """)
    logs = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM sensor_logs")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(fsr), MAX(fsr), AVG(stability_score) FROM sensor_logs")
    stats = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM sensor_logs WHERE injury_risk=TRUE")
    risk_count = cursor.fetchone()[0]

    cursor.close(); conn.close()
    return render_template('history.html', logs=logs, total=total,
                           stats=stats, risk_count=risk_count, user=current_user)

# ── Settings ──
@app.route('/settings', methods=['GET','POST'])
@login_required
def settings():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'change_password':
            current_pw = request.form.get('current_password', '')
            new_pw     = request.form.get('new_password', '')
            confirm_pw = request.form.get('confirm_password', '')

            conn   = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE id=%s", (current_user.id,))
            row = cursor.fetchone()

            if not check_password_hash(row[0], current_pw):
                flash('Current password is incorrect.', 'error')
            elif len(new_pw) < 6:
                flash('New password must be at least 6 characters.', 'error')
            elif new_pw != confirm_pw:
                flash('New passwords do not match.', 'error')
            else:
                cursor.execute(
                    "UPDATE users SET password=%s WHERE id=%s",
                    (generate_password_hash(new_pw), current_user.id)
                )
                conn.commit()
                flash('Password changed successfully.', 'success')

            cursor.close(); conn.close()

        elif action == 'delete_account':
            confirm = request.form.get('confirm_delete', '')
            if confirm == current_user.username:
                conn   = get_db()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE id=%s", (current_user.id,))
                cursor.execute("DELETE FROM login_logs WHERE username=%s", (current_user.username,))
                conn.commit()
                cursor.close(); conn.close()
                logout_user()
                flash('Account deleted.', 'info')
                return redirect('/login')
            else:
                flash('Username confirmation did not match.', 'error')

        return redirect('/settings')

    return render_template('settings.html', user=current_user)

# =========================
# SENSOR API
# =========================
@app.route('/sensor/', methods=['POST'])
def sensor():
    global latest_fsr_value, pressure_state, step_count, walking_status
    global step_detected, pressure_history, fall_detected, ml_prediction
    global gait_status, injury_risk, heavy_pressure_counter, movement_intensity
    global fall_confidence, stability_score, email_sent
    global ax, ay, az, gx, gy, gz, sequence_buffer, lstm_prediction, latest_data

    data      = request.json
    latest_data = data

    latest_fsr_value = data['fsr']
    ax = data['ax']; ay = data['ay']; az = data['az']
    gx = data['gx']; gy = data['gy']; gz = data['gz']

    accel_magnitude = abs(ax) + abs(ay) + abs(az)
    gyro_magnitude  = abs(gx) + abs(gy) + abs(gz)
    movement_intensity = accel_magnitude + gyro_magnitude

    current_frame = [latest_fsr_value, ax, ay, az, gx, gy, gz,
                     movement_intensity, stability_score]
    sequence_buffer.append(current_frame)
    if len(sequence_buffer) > 20:
        sequence_buffer.pop(0)

    stability_score = 100
    if movement_intensity > 20000: stability_score -= 20
    if gyro_magnitude > 1000:      stability_score -= 30
    if fall_confidence > 70:       stability_score -= 40
    if stability_score < 0:        stability_score = 0

    pressure_history.append(latest_fsr_value)
    if len(pressure_history) > 10: pressure_history.pop(0)

    input_data = pd.DataFrame([[latest_fsr_value]], columns=['fsr'])
    prediction = model.predict(input_data)
    ml_prediction = prediction[0]
    print("AI Prediction:", ml_prediction)

    if len(sequence_buffer) == 20:
        try:
            sequence_array   = np.array(sequence_buffer)
            scaler           = MinMaxScaler()
            scaled_sequence  = scaler.fit_transform(sequence_array)
            X_input          = np.expand_dims(scaled_sequence, axis=0)
            predicted_class  = 0
            lstm_prediction  = "Normal" if predicted_class == 0 else "Instability"
            print("LSTM Prediction:", lstm_prediction)
        except Exception as e:
            print("LSTM ERROR:", e)

    fall_confidence = 0
    if latest_fsr_value > 3000:                                fall_confidence += 30
    if accel_magnitude > 25000:                                fall_confidence += 35
    if gyro_magnitude > 800:                                   fall_confidence += 25
    if latest_fsr_value < 200 and max(pressure_history) > 3000: fall_confidence += 20
    if fall_confidence > 100: fall_confidence = 100

    gait_status = "Abnormal" if latest_fsr_value > 3500 else "Normal"

    if latest_fsr_value > 3000: heavy_pressure_counter += 1
    else:                        heavy_pressure_counter  = 0

    if heavy_pressure_counter >= 5:
        injury_risk = True; gait_status = "INJURY RISK"
    else:
        injury_risk = False

    fall_detected = False
    if fall_confidence >= 70:
        fall_detected = True; gait_status = "FALL DETECTED"
        injury_risk   = True
        print("FALL DETECTED!")

    if fall_detected and not email_sent:
        try:
            msg = MIMEMultipart()
            msg['From']    = EMAIL_ADDRESS
            msg['To']      = ALERT_RECEIVER
            msg['Subject'] = "EMERGENCY FALL DETECTED"
            body = f"""Emergency detected by Smart Insole AI.
Fall Confidence: {fall_confidence}%
FSR Pressure: {latest_fsr_value}
Stability Score: {stability_score}
Movement Intensity: {movement_intensity}
Gait Status: {gait_status}
Immediate attention may be required."""
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print("EMAIL ALERT SENT!")
            email_sent = True
        except Exception as e:
            print("EMAIL ERROR:", e)

    if fall_confidence < 20: email_sent = False

    if latest_fsr_value < 150:   pressure_state = "No Pressure"
    elif latest_fsr_value < 700: pressure_state = "Light Pressure"
    else:                        pressure_state = "Heavy Pressure"

    STEP_THRESHOLD = 700
    if latest_fsr_value > STEP_THRESHOLD and not step_detected:
        step_count += 1; step_detected = True; walking_status = "Walking"
    elif latest_fsr_value < 300:
        step_detected = False; walking_status = "Idle"

    sensor_buffer.append({
        "fsr": latest_fsr_value, "prediction": ml_prediction,
        "injury_risk": injury_risk, "fall_confidence": fall_confidence,
        "stability_score": stability_score, "movement_intensity": movement_intensity,
        "ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz
    })

    return jsonify({"status": "received", "injury_risk": injury_risk})

# =========================
# FRONTEND API
# =========================
@app.route('/get_data')
def get_data():
    return jsonify({
        'ax': ax, 'ay': ay, 'az': az,
        'gx': gx, 'gy': gy, 'gz': gz,
        'fsr': latest_fsr_value, 'state': pressure_state,
        'steps': step_count, 'walking': walking_status,
        'prediction': ml_prediction, 'gait': gait_status,
        'injury_risk': injury_risk, 'fall': fall_detected,
        'fall_confidence': fall_confidence,
        'stability_score': stability_score,
        'movement_intensity': movement_intensity,
        'lstm_prediction': lstm_prediction,
    })

# =========================
# START SERVER
# =========================
# threading.Thread(target=database_worker, daemon=True).start()
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
