import sys
sys.path.append('C:/Users/HP/AppData/Local/Programs/Python/Python312/Lib/site-packages')
import cv2
from random import randrange

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to draw rectangles around detected faces
def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (randrange(255),randrange(255),0), 2)

# Main function
def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If frame is not captured properly, exit
        if not ret:
            break

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Draw rectangles around detected faces
        draw_faces(frame, faces)

        # Display the frame
        cv2.imshow('Face Detection', frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
