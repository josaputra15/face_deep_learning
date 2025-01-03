import cv2 
import os
import numpy as np


# Global variables
face_cascade = cv2.CascadeClassifier("/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
camera = cv2.VideoCapture(0)



# 1: Capture
def capture_images(name):
    if not os.path.exists("training_data"):
        os.makedirs("training_data")

    count = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(100, 100))

        for x, y, w, h in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            filename = f"training_data/{name}_{count}.jpg"
            cv2.imwrite(filename, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Images", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:  
            break

    print(f"Captured {count} images for {name}")
    camera.release()
    cv2.destroyAllWindows()

#2: Train
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
    with open("label_map2.txt", "w") as f:
        for name, label in label_map.items():
            f.write(f"{name}:{label}\n")

    print("Training completed. Model saved.")

# 3:Recognitions
def recognizer_box_with_landmarks(frame):
    if not os.path.exists("face_recognizer.yml") or not os.path.exists("label_map2.txt"):
        print("No trained model found. Please train the recognizer first!")
        return

    recognizer.read("face_recognizer.yml")
    with open("label_map2.txt", "r") as f:
        label_map = {int(line.split(":")[1]): line.split(":")[0] for line in f.readlines()}

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

    for x, y, w, h in faces:
        if w < 50 or h < 50: 
            continue

        face = gray_frame[y:y+h, x:x+w]
        face = cv2.equalizeHist(face)

        label, confidence = recognizer.predict(face)

        if confidence > 50: 
            name = "Unknown"
        else:
            name = label_map[label]

        # boundingbox display
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        # rect = dlib.rectangle(x, y, x + w, y + h)
        # landmarks = landmark_predictor(gray_frame, rect)

        # #biometric
        # for i in range(68):
        #     point = (landmarks.part(i).x, landmarks.part(i).y)
        #     cv2.circle(frame, point, 2, (0, 255, 255), -1)

# Main
def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        recognizer_box_with_landmarks(frame)
        cv2.imshow("Biometric", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()

   
# Unified Entry Point
if __name__ == "__main__":
    print("Choose an option:")
    print("1: Capture Images")
    print("2: Train Recognizer")
    print("3: Real-Time Recognition with Biometric Dots")

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



#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------#--------------

