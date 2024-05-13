#import necessary libraries
from gpiozero import DigitalOutputDevice
from gpiozero import DigitalInputDevice
from gpiozero import PWMOutputDevice
from time import sleep

ENCODER_PIN = 18  # Replace with the actual GPIO pin connected to the encoder
WHEEL_CIRCUMFERENCE_CM = 16.65  # wheel circumference constant in centimeters calculated by 2*pi*2.65  where 2.65 is the radius of the wheels in centimeters
encoder = DigitalInputDevice(ENCODER_PIN) # Initialize the encoder

# Initialize variables
rotations = 0
total_distance_cm = 0

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

def handle_rotation():
    """Handles a single rotation event detected by the encoder."""
    global rotations, total_distance_cm  # Access global variables for updates

    rotations += 1 # Increment the rotation count

    if rotations == 20:
        rotations = 0 # Reset the rotation count after 20 rotations
    else:
        total_distance_cm += WHEEL_CIRCUMFERENCE_CM / 20  # Calculate partial distance

encoder.when_activated = handle_rotation  # Assign the handle_rotation function to the encoder

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
    while True:
        user_input = input("Enter 'f' for forward motion (or 'q' to quit): ") # receives user input for vehicle's motion
        if user_input.lower() == 'f':
            """Vehicle move forward"""
            forward()
            en1.value = 0.4 
            en2.value = 0.4  
            print("Motor moving forward")

        elif user_input.lower() == 'r':
            """Vehicle turn right"""
            right()
            en1.value = 1.0  
            en2.value = 1.0  
            print("Motor moving right")

        elif user_input.lower() == 'l':
            """Vehicle turn left"""
            left()
            en1.value = 1.0  
            en2.value = 1.0  
            print("Motor moving left")

        elif user_input.lower() == '90l':
            """Vehicle turn 90 degree to the left"""
            left()
            en1.value = 0.85  
            en2.value = 0.85  
            sleep(0.47)
            print("Motor moving left")
            stop()

        elif user_input.lower() == '75l':
            """Vehicle turn 75 degree to the left"""
            left()
            en1.value = 0.85  
            en2.value = 0.85 
            sleep(0.42)
            print("Motor moving left")
            stop()

        elif user_input.lower() == '45l':
            """Vehicle turn 45 degree to the left"""
            left()
            en1.value = 0.85 
            en2.value = 0.85  
            sleep(0.27)
            print("Motor moving left")
            stop()

        elif user_input.lower() == '90r':
            """Vehicle turn 90 degree to the right"""
            right()
            en1.value = 0.85  
            en2.value = 0.85 
            print("Motor moving right")
            sleep(0.47)
            stop()

        elif user_input.lower() == '75r':
            """Vehicle turn 75 degree to the right"""
            right()
            en1.value = 0.85 
            en2.value = 0.85 
            print("Motor moving right")
            sleep(0.42)
            stop()

        elif user_input.lower() == '45r':
            """Vehicle turn 45 degree to the right"""
            right()
            en1.value = 0.85  
            en2.value = 0.85  
            print("Motor moving right")
            sleep(0.27)
            stop()

        elif user_input.lower() == 'b':
            """Vehicle move backward"""
            backward()
            en1.value = 1.0  
            en2.value = 1.0 
            print("Motor moving backward")

        elif user_input.lower() == 'h2l':
            """Vehicle moving forward from high speed to low speed"""
            forward()
            en1.value = 1.0  
            en2.value = 1.0  
            sleep(2)
            en1.value = 0.6  
            en2.value = 0.6  
            sleep(2)
            en1.value = 0.3  
            en2.value = 0.3  
            sleep(2)
            en1.value = 0.0 
            en2.value = 0.0  

        elif user_input.lower() == 'l2h':
            """Vehicle moving forward from low speed to high speed"""
            forward()
            en1.value = 0.4  
            en2.value = 0.4  
            sleep(0.5)
            en1.value = 0.3 
            en2.value = 0.3  
            sleep(2)
            en1.value = 0.6
            en2.value = 0.6  
            sleep(2)
            en1.value = 1.0  
            en2.value = 1.0 
            sleep(2)
            en1.value = 0.0  
            en2.value = 0.0 

        elif user_input.lower() == 's':
            """Vehicle stop"""
            stop()

        else:
            """Prints error message; User enters invalid input"""
            print("Invalid input. Enter 'f' for forward motion or 'q' to quit.")


except KeyboardInterrupt:
    """Vehicle stop to display distance travelled"""
    print("Keyboard interrupt detected. Stopping motors.")
    print(f"Rotations: {rotations}, Total Distance (cm): {total_distance_cm / 15.65:.2f}")
    stop()


finally:
    """Clears off GPIO pin"""
    in1.close()
    in2.close()
    in3.close()
    in4.close()
    en1.close()
    en2.close()
    encoder.close()
    print("Motors stopped. Program terminated.")
