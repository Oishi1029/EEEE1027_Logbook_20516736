import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.resolution = (192,108)
picam2.framerate = 20
picam2.preview_configuration.main.size = (192,108)  
picam2.preview_configuration.main.format = "RGB888"  
picam2.preview_configuration.align()  
picam2.configure("preview")  
picam2.start()
time.sleep(2)
def display_hsv_values(frame):
    # Convert the frame from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    hue_channel, saturation_channel, value_channel = cv2.split(hsv_frame)

    # Display the HSV channels
    cv2.imshow('Hue Channel', hue_channel)
    cv2.imshow('Saturation Channel', saturation_channel)
    cv2.imshow('Value Channel', value_channel)

# Open the default camera (usually the webcam)
#cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()


    # Display the frame
    cv2.imshow('Original', frame)

    # Display HSV values
    display_hsv_values(frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
#cap.release()
cv2.destroyAllWindows()
