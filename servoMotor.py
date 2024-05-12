from gpiozero import Servo
from time import sleep

# Define the GPIO pin for the servo
myGPIO = 17  # You can change this to any available GPIO pin

# Create a Servo object
servo = Servo(myGPIO)

servo.mid()  # Move the servo to the middle position
print("Mid position")
sleep(0.5)

servo.min()  # Move the servo to the middle position
print("Mid position")
sleep(0.5)

servo.max()
sleep(0.5)
