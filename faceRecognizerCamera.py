import cv2
import threading
from queue import Queue
import os

# Queue to pass frames from worker threads to the main thread
frame_queue = Queue()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/haarcascade_frontalface_alt2.xml')

# Load face recognizer and label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/face_recognizer.yml')

# Load label map from file
with open('/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/label_map.txt', 'r') as f:
    label_map = {int(line.split(",")[0]): line.split(",")[1].strip() for line in f.readlines()}

def process_video(video_path, queue):
    """
    Process a video in a separate thread.
    Reads frames and adds them to the queue.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Add the frame to the queue with its video name
        queue.put((os.path.basename(video_path), frame))
    cap.release()
    # Signal the main thread that this video is done
    queue.put((os.path.basename(video_path), None))

def recognizer_box_with_landmarks(frame, label_map):
    """
    Process a frame for face detection, recognition, and biometric landmarks.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))

    for x, y, w, h in faces:
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.equalizeHist(face)

        label, confidence = recognizer.predict(face)

        if confidence > 50:
            name = "Unknown"
        else:
            name = label_map[label]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def display_frames(queue, video_count):
    """
    Display frames from the queue in separate windows.
    """
    finished_videos = 0
    while finished_videos < video_count:
        # Retrieve frames from the queue
        video_name, frame = queue.get()
        if frame is None:  # Exit signal for one video
            finished_videos += 1
            cv2.destroyWindow(video_name)  # Close the window for this video
            continue

        # Recognize and annotate faces
        frame = recognizer_box_with_landmarks(frame, label_map)

        # Display the frame
        cv2.imshow(video_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    # List of video paths
    video_paths = [
        "/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/josep.mp4", 
        "/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/josep2.mp4",
        "/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/joko1.mp4",
        "/Users/josephsaputra/Desktop/comp128 joseph/face_deep_learning/joko2.mp4",
    ]

    # Start threads for each video
    threads = []
    for video_path in video_paths:
        thread = threading.Thread(target=process_video, args=(video_path, frame_queue))
        threads.append(thread)
        thread.start()

    # Display frames in the main thread
    display_frames(frame_queue, len(video_paths))

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()


