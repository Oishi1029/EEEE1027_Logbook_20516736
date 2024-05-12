import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep
picam2 = Picamera2()
picam2.resolution = (192,108)
picam2.framerate = 35
picam2.preview_configuration.main.size = (192,108)  
picam2.preview_configuration.main.format = "RGB888"  
picam2.preview_configuration.align()  
picam2.configure("preview")  
picam2.start()
sleep(2)

# Initialize the camera
#camera = PiCamera2()
#camera.start_preview()

# Capture an image
while 1:
        
    image = picam2.capture_array()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

    # If circles are detected, draw them on the image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print("Circle")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    else:
        print("None")

    # Display the image with detected circles
    cv2.imshow("Circles", image)
    if cv2.waitKey(10) & 0xFF == ord('q'): 
            cv2.destroyAllWindows() 
cv2.destroyAllWindows()

# Release the camera resources
#camera.stop_preview()
picam2.close()
