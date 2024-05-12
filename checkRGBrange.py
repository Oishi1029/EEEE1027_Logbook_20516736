import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep
# Open the camera
#cap = cv2.VideoCapture(0)

picam2 = Picamera2()
picam2.resolution = (192,108)
picam2.framerate = 20
picam2.preview_configuration.main.size = (192,108)  
picam2.preview_configuration.main.format = "RGB888"  
picam2.preview_configuration.align()  
picam2.configure("preview")  
picam2.start()
sleep(2)


# Define the color range for detection (in RGB)
#blue
#lower_color = np.array([0, 0, 65])  # Lower bound for the color (adjust as needed)
#upper_color = np.array([100, 100, 255])  # Upper bound for the color (adjust as needed)

#green
#lower_color = np.array([0, 100, 0])  # Lower bound for the color (adjust as needed)
#upper_color = np.array([100, 255, 100])  # Upper bound for the color (adjust as needed)


#red
#lower_color = np.array([100, 0, 0])  # Lower bound for the color (adjust as needed)
#upper_color = np.array([255, 100, 100])  # Upper bound for the color (adjust as needed)

#black
#lower_color = np.array([0, 0, 0])  # Lower bound for the color (adjust as needed)
#upper_color = np.array([30, 30, 30])  # Upper bound for the color (adjust as needed)

#yellow
lower_color = np.array([150, 150, 0])  # Lower bound for the color (adjust as needed)
upper_color = np.array([255, 255, 100])  # Upper bound for the color (adjust as needed)

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()
    #ret, frame = cap.read()
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Threshold the image to get only the specified color range
    mask = cv2.inRange(frame_rgb, lower_color, upper_color)
    
    # Bitwise AND operation to extract the detected color regions
    detected_color = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
    
    # Convert back to BGR for display
    detected_color_bgr = cv2.cvtColor(detected_color, cv2.COLOR_RGB2BGR)
    
    # Display the original and detected color images
    cv2.imshow('Original Image', frame)
    cv2.imshow('Detected Color', detected_color_bgr)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
picam2.release()
cv2.destroyAllWindows()
