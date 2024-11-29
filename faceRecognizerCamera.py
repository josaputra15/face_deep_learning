import cv2

# Load the face detection model
face_ref = cv2.CascadeClassifier("face_algortim_frontal_opencv.xml")
camera = cv2.VideoCapture(0)

# Face detection function
def face_detection(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(grey_frame, scaleFactor=1.1, minSize=(100, 100), minNeighbors=5)
    return faces

# Recognizer box function
def recognizer_box(frame):
    for x, y, w, h in face_detection(frame):
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        # Display name below the rectangle
        text_position = (x, y + h + 30)  # Position below the box
        cv2.putText(frame, "person", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Function to close the window
def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

# Main function
def main():
    while True:
        _, frame = camera.read()
        recognizer_box(frame)
        cv2.imshow("Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

# Run the main function
if __name__ == '__main__':
    main()
