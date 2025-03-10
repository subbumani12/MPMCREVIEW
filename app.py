from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import pyotp  # For Google Authenticator–style TOTP

app = Flask(__name__)

# -------------------
# 1) Facial Recognition Setup
# -------------------

# Relative paths for model files (must be in same directory as app.py)
model_path = os.path.join(os.getcwd(), "face_recognizer.yml")
label_map_path = os.path.join(os.getcwd(), "label_map.npy")


# Check if files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} does not exist. Please copy it into the project directory.")
if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"{label_map_path} does not exist. Please copy it into the project directory.")

# Load the trained model and label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
label_map = np.load(label_map_path, allow_pickle=True).item()
reverse_label_map = {v: k for k, v in label_map.items()}

# Initialize face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------
# 2) Google Authenticator TOTP Setup
# -------------------
# Replace with your own secret key if needed
TOTP_SECRET = "JBSWY3DPEHPK3PXP"

@app.route('/authcode')
def authcode():
    """
    Returns a JSON with the current TOTP code.
    """
    totp = pyotp.TOTP(TOTP_SECRET)
    return jsonify({"authcode": totp.now()})


# -------------------
# 3) Video Stream Generation (Facial Recognition)
# -------------------
def gen_frames():
    """
    Captures frames from the webcam, runs facial recognition,
    and yields an MJPEG stream.
    """
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))
            label, confidence = recognizer.predict(face_img)

            # Adjust threshold as needed
            if confidence > 70:
                name = "Unauthorized"
            else:
                name = reverse_label_map.get(label, "Unknown")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{name} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255) if name == "Unauthorized" else (0, 255, 0),
                        2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # MJPEG magic
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    """
    Returns the video stream with bounding boxes and labels
    for recognized/unrecognized faces.
    """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------
# 4) Routes for Pages
# -------------------

@app.route('/')
def home():
    """
    Renders the index (login) page.
    """
    return render_template('index.html')


@app.route('/auth')
def auth_page():
    """
    Renders the TOTP auth code page.
    """
    return render_template('auth.html')


@app.route('/stream')
def stream_page():
    """
    Renders the live camera stream page.
    """
    return render_template('stream.html')


# -------------------
# 5) Run the App
# -------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
