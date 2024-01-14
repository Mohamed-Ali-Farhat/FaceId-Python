# Import the OpenCV library
import cv2
# Import the SimpleFacerec class from the simple_facerec module
from simple_facerec import SimpleFacerec

# Create an instance of the SimpleFacerec class
sfr = SimpleFacerec()
# Load face encodings from a folder containing images
sfr.load_encoding_images("images/")

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Start an infinite loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is not empty
    if not ret or frame is None:
        print("Error: Could not read frame from the camera.")
        break

    # Detect faces in the current frame
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    # Iterate through the detected faces and their corresponding names
    for face_loc, name in zip(face_locations, face_names):
        # Extract coordinates of the detected face
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Draw the name above the face
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Display the frame with drawn rectangles and names
    cv2.imshow("Frame", frame)

    # Check for the 'Esc' key to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
