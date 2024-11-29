import cv2
import os
import numpy as np

# Global variables
face_cascade = cv2.CascadeClassifier("face_algortim_frontal_opencv.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
camera = cv2.VideoCapture(0)

# 1 Capture Images for Training
def capture_images(name):
    if not os.path.exists("training_data"):
        os.makedirs("training_data")

    count = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for x, y, w, h in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            filename = f"training_data/{name}_{count}.jpg"
            cv2.imwrite(filename, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Images", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:  # Capture 20 images or quit with 'q'
            break

    print(f"Captured {count} images for {name}")
    camera.release()
    cv2.destroyAllWindows()

# 2 Train the Recognizer
def train_recognizer():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists("training_data"):
        print("No training data found. Please capture images first!")
        return

    for filename in os.listdir("training_data"):
        if filename.endswith(".jpg"):
            name = filename.split("_")[0]
            if name not in label_map:
                label_map[name] = current_label
                current_label += 1

            filepath = os.path.join("training_data", filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_map[name])

    recognizer.train(faces, np.array(labels))
    recognizer.save("face_recognizer.yml")
    with open("label_map.txt", "w") as f:
        for name, label in label_map.items():
            f.write(f"{name}:{label}\n")

    print("Training completed. Model saved.")

# Step 3: Real-Time Recognition
def recognizer_box(frame):
    if not os.path.exists("face_recognizer.yml") or not os.path.exists("label_map.txt"):
        print("No trained model found. Please train the recognizer first!")
        return

    recognizer.read("face_recognizer.yml")
    with open("label_map.txt", "r") as f:
        label_map = {int(line.split(":")[1]): line.split(":")[0] for line in f.readlines()}

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for x, y, w, h in faces:
        face = gray_frame[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 100:  # Confidence threshold
            name = label_map[label]
        else:
            name = "Unknown"

        # Draw rectangle and display name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Main Method for Real-Time Recognition
def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        recognizer_box(frame)
        cv2.imshow("Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Unified Entry Point
if __name__ == "__main__":
    print("Choose an option:")
    print("1: Capture Images")
    print("2: Train Recognizer")
    print("3: Real-Time Recognition")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        name = input("Enter the name of the person: ").strip().lower()
        capture_images(name)
    elif choice == "2":
        train_recognizer()
    elif choice == "3":
        main()
    else:
        print("Invalid choice. Exiting...")
