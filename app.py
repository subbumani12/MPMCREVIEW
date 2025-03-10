from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)

data_path = "face_data"
os.makedirs(data_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_face', methods=['POST'])
def upload_face():
    data = request.get_json()
    name = data.get('name')
    image_data = data.get('image')

    if not name or not image_data:
        return jsonify({"error": "Missing name or image"}), 400

    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    samples_collected = 0
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))
        file_path = os.path.join(data_path, f"{name}_{samples_collected}.jpg")
        cv2.imwrite(file_path, face)
        samples_collected += 1
    
    return jsonify({"message": "Face saved successfully", "count": samples_collected})

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({"error": "Missing image data"}), 400

    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))
        results.append({"name": "Face Detected"})
    
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
