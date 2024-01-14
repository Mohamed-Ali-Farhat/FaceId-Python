import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        # Print the number of encoding images found
        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            # Convert image from BGR to RGB color space (required by face_recognition)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)


            # Get encoding for the current image
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        
        
        print("Encoding images loaded")

    
    
    
    def detect_known_faces(self, frame):
        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convert the image from BGR to RGB color space
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Initialize an empty list to store names of detected faces
        face_names = []
        
        # Iterate through each detected face
        for face_encoding in face_encodings:
            # Check if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # If a match is found, set the name to the corresponding known face name
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name) #ya nkown wala esmo ken fama best_match_index krib lel tssawer eli aandi f images

        # Convert face locations to a numpy array and adjust coordinates with frame resizing
        #besh taamel kal cadre lahmer
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        # Return the adjusted face locations and names
        return face_locations.astype(int), face_names