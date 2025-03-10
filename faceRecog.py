import cv2
import numpy as np
import os

# Path to store face data
data_path = "face_data"
os.makedirs(data_path, exist_ok=True)

# Function to collect face samples
def collect_faces(name, num_samples=50):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    print("Collecting samples for", name)
    samples_collected = 0

    while samples_collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))

            file_path = os.path.join(data_path, f"{name}_{samples_collected}.jpg")
            cv2.imwrite(file_path, face)
            samples_collected += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to train the recognizer
def train_recognizer():
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
    print("Model trained successfully")
    return recognizer, label_map

# Function to recognize faces
def recognize_faces(recognizer, label_map):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    reverse_label_map = {v: k for k, v in label_map.items()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))

            label, confidence = recognizer.predict(face)
            print(f"Predicted label: {label}, Confidence: {confidence}")  # Debug print

            if confidence > 70:  # Adjusted threshold for unauthorized access
                name = "Unauthorized"
            else:
                name = reverse_label_map.get(label, "Unknown")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if name == "Unauthorized" else (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Workflow
if __name__ == "__main__":
    print("1. Collect Faces\n2. Train Model\n3. Recognize Faces")
    choice = input("Enter your choice: ")

    if choice == "1":
        name = input("Enter the name of the person: ")
        collect_faces(name)
    elif choice == "2":
        recognizer, label_map = train_recognizer()
        # Save the trained model
        recognizer.write("face_recognizer.yml")
        np.save("label_map.npy", label_map)
    elif choice == "3":
        # Load the trained model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face_recognizer.yml")
        label_map = np.load("label_map.npy", allow_pickle=True).item()
        recognize_faces(recognizer, label_map)
    else:
        print("Invalid choice!")
