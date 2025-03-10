import cv2
import numpy as np
import os
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to store face data
data_path = "face_data"
os.makedirs(data_path, exist_ok=True)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to process uploaded face images
@app.route('/upload_face', methods=['POST'])
def upload_face():
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')

        if not name or not image_data:
            return jsonify({"error": "Missing name or image"}), 400

        # Decode Base64 image safely
        try:
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400

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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to train the recognizer
@app.route('/train', methods=['POST'])
def train_recognizer():
    try:
        if not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            return jsonify({"error": "OpenCV does not support LBPHFaceRecognizer"}), 500

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        labels = []
        label_map = {}
        current_label = 0

        for file in os.listdir(data_path):
            if file.endswith(".jpg"):
                label = file.split("_")[0]
                if label not in label_map:
                    label_map[label] = current_label
                    current_label += 1

                img_path = os.path.join(data_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_map[label])

        recognizer.train(faces, np.array(labels))
        recognizer.write("face_recognizer.yml")
        np.save("label_map.npy", label_map)

        return jsonify({"message": "Model trained successfully", "labels": label_map})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to recognize faces from an uploaded image
@app.route('/recognize', methods=['POST'])
def recognize_faces():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "Missing image data"}), 400

        if not os.path.exists("face_recognizer.yml"):
            return jsonify({"error": "Face recognition model not trained yet"}), 500

        # Load trained model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face_recognizer.yml")
        label_map = np.load("label_map.npy", allow_pickle=True).item()
        reverse_label_map = {v: k for k, v in label_map.items()}

        # Decode Base64 image safely
        try:
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        results = []
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))
            
            label, confidence = recognizer.predict(face)
            name = "Unauthorized" if confidence > 70 else reverse_label_map.get(label, "Unknown")

            results.append({"name": name, "confidence": confidence})

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
