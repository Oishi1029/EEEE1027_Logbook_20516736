#https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/

import numpy as np
import cv2
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.resolution = (192, 108)
picam2.framerate = 30
picam2.preview_configuration.main.size = (192, 108)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
sleep(2)

# Initialize the camera
#cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    frame = picam2.capture_array()
    #ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the grayscale image
    _, threshold = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    # Detect contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and draw them on the frame
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)

        # Draw the contours on the frame
        cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)

        # Get the coordinates of the contour
        n = approx.ravel()
        i = 0
        coords = []
        for j in n:
            if i % 2 == 0:
                x = n[i]
                y = n[i + 1]
                coords.append((x, y))
            i += 1

        # Draw the coordinates on the frame
        for coord in coords:
            cv2.circle(frame, coord, 5, (0, 255, 0), -1)
            cv2.putText(frame, str(coord), (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
picam2.release()
cv2.destroyAllWindows()
