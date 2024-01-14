Image Recognition Project
Overview
This project is focused on image recognition using Python, with a special emphasis on the OpenCV library (cv2). The primary functionality revolves around face recognition, and the main components of the project are organized in separate files.

Files
simple_facerec.py: This file contains the core functionality of the image recognition system. Here's an overview of the functions:

load_encoding_images: Loads images from the "images" folder for encoding.
detect_unknown_faces: Implements the face detection function using OpenCV.
image_comparison.py: This file serves as a test suite for the encoding functions. It includes tests to validate the functionality of the encoding process.

main.py: The main file that orchestrates the entire image recognition process. It integrates the functions from simple_facerec.py to perform the complete recognition workflow.

Usage
To utilize this project, follow these steps:




Ensure you have Python installed on your machine.
Install the required dependencies, including OpenCV with this command
pip install opencv-python

Populate the "images" folder with the images you want to use for encoding.
Run main.py to initiate the image recognition process.




Feel free to explore and adapt the code to suit your specific use case. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.
