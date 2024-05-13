
# Import necessary libraries
import cv2
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from time import sleep

# Pin configuration
in1_pin = 24
in2_pin = 23
in3_pin = 27
in4_pin = 22
en1_pin = 12
en2_pin = 13

# Initialize output pins for controlling vehicle's direction
in1 = DigitalOutputDevice(in1_pin)
in2 = DigitalOutputDevice(in2_pin)
in3 = DigitalOutputDevice(in3_pin)
in4 = DigitalOutputDevice(in4_pin)

# Initialize PWM output pins for controlling vehicle's speed
en1 = PWMOutputDevice(en1_pin)
en2 = PWMOutputDevice(en2_pin)

# Initialize Camera Module
picam2 = Picamera2()
picam2.resolution = (192, 108)
picam2.framerate = 20
picam2.preview_configuration.main.size = (192, 108)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
sleep(2)

def forward():
    """Configure direction of vehicle to forward"""
    in1.on()
    in2.off()
    in3.on()
    in4.off()

def right():
    """Configure direction of vehicle to right"""
    in1.on()
    in2.off()
    in3.off()
    in4.on()

def left():
    """Configure direction of vehicle to left"""
    in1.off()
    in2.on()
    in3.on()
    in4.off()

def stop():
    """Stops the robot"""
    in1.on()
    in2.on()
    in3.on()
    in4.on()
    en1.value = 0.0
    en2.value = 0.0

def backward():
    """Configure direction of vehicle to backward"""
    in1.off()
    in2.on()
    in3.off()
    in4.on()

try:
    while (True):
        crop_img = picam2.capture_array() # capture video frame
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) # Convert video frame to grayscale
        blur = cv2.GaussianBlur(gray, (5, 5), 0) # apply Gaussian blur to video frame
        ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV) # Apply a threshold to the blurred image
        mask = cv2.erode(thresh, None, iterations=2)  # Reduce noise with erosion
        mask = cv2.dilate(mask, None, iterations=2) # Expand the detected line
        contours, hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE) # Find contours (outlines) in the mask

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea) # Find the largest contour
            M = cv2.moments(c) # Calculate image moments for center calculation
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) # Calculate the x-coordinate of the line's center

            else:
                cx = 0 # Handle cases where the line might be undetectable

            if cx <= 40:
                """Vehicle turn left"""
                print("Turn Left!")
                left()
                en1.value = 0.75
                en2.value = 0.75  

            if cx < 150 and cx > 40:
                """Vehicle move forward"""
                print("On Track!")
                forward()
                en1.value = 0.4  
                en2.value = 0.4  

            if cx >= 150:
                """Vehicle turn right"""
                print("Turn Right")
                right()
                en1.value = 0.75  
                en2.value = 0.75  
        else:
            """Vehicle stop"""
            stop()
            print("I don't see the line")

        cv2.imshow('frame', crop_img) # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Exit the loop when 'q' key is pressed
            break

except KeyboardInterrupt:
    """Keyboard interrupt detected, vehicle perform stop"""
    print("Keyboard interrupt detected. Stopping motors.")
    stop()

finally:
    """Clean up GPIO pin"""
    in1.close()
    in2.close()
    in3.close()
    in4.close()
    en1.close()
    en2.close()
    print("Motors stopped. Program terminated.")
