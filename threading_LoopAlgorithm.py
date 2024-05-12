# debugged event.set() for other colour

import numpy as np
import cv2
import time
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from time import sleep
import threading
from keras.models import load_model

num_colors = int(input("How many preferred colors do you have in mind? "))
preferred_colors = []

for i in range(num_colors):
    color = input("Enter color number " + str(i + 1) + ": ").lower()
    preferred_colors.append(color)

modelRed = load_model("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/RED/converted_keras (28)/keras_model.h5",compile=False)
class_namesRed = open("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/RED/converted_keras (28)/labels.txt", "r").readlines()

modelGreen = load_model("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/GREEN/converted_keras (29)/keras_model.h5",compile=False)
class_namesGreen = open("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/GREEN/converted_keras (29)/labels.txt", "r").readlines()

modelBlue = load_model("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/BLUE/converted_keras (30)/keras_model.h5",compile=False)
class_namesBlue = open("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/BLUE/converted_keras (30)/labels.txt", "r").readlines()

modelPurple = load_model("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/PURPLE/converted_keras (31)/keras_model.h5",compile=False)
class_namesPurple = open("/home/dean/Documents/Week3 Training Image for 700/KERAS COLOURED MODEL/PURPLE/converted_keras (31)/labels.txt", "r").readlines()

# initialize RGB range
red_lower = np.array([65, 0, 5], np.uint8)
red_upper = np.array([210, 45, 50], np.uint8)

green_lower = np.array([5, 130, 75], np.uint8)
green_upper = np.array([20, 141, 90], np.uint8)

blue_lower = np.array([30, 45, 80], np.uint8)
blue_upper = np.array([60, 90, 120], np.uint8)

# Add black color detection
black_lower = np.array([7, 10, 10], np.uint8)
black_upper = np.array([40, 45, 50], np.uint8)

yellow_lower = np.array([205, 170, 5], np.uint8)
yellow_upper = np.array([230, 210, 50], np.uint8)

kernel = np.ones((5, 5), "uint8")

redSymbols = ["Stop", "Measure Distance"]
"""""
red_mask = cv2.inRange(rgbFrame, red_lower, red_upper)
res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

green_mask = cv2.inRange(rgbFrame, green_lower, green_upper)
res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

blue_mask = cv2.inRange(rgbFrame, blue_lower, blue_upper)
res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

black_mask = cv2.inRange(rgbFrame, black_lower, black_upper)
res_black = cv2.bitwise_and(imageFrame, imageFrame, mask=black_mask)

yellow_mask = cv2.inRange(rgbFrame, yellow_lower, yellow_upper)
res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)
"""


# Define GPIO pins
in1_pin = 24
in2_pin = 23
in3_pin = 27
in4_pin = 22
en1_pin = 12
en2_pin = 13

Kp = 0.0060
Ki = 0.0
Kd = 0.0

setpoint = 102.5  # Desired distance from the line
# setpoint = 80  # Desired distance from the line
prev_error = 0
integral = 0
derivative = 0

look_ahead_frames = 5  # Number of frames to look ahead
look_ahead_setpoint = 102.5  # Desired distance from the line for look-ahead

# Initialize GPIO devices
in1 = DigitalOutputDevice(in1_pin)
in2 = DigitalOutputDevice(in2_pin)
in3 = DigitalOutputDevice(in3_pin)
in4 = DigitalOutputDevice(in4_pin)
en1 = PWMOutputDevice(en1_pin)
en2 = PWMOutputDevice(en2_pin)

picam2 = Picamera2()
picam2.resolution = (192, 108)
picam2.framerate = 35
picam2.preview_configuration.main.size = (192, 108)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
sleep(2)

# Initialize color sets
detected_colors = set()


def backward():
    in1.off()  # blue dc motor
    in2.on()
    in3.off()  # white dc motor
    in4.on()
    # en1.value = 0.34  # Set motor speed (0 to 1)
    # en2.value = 0.34  # Set motor speed (0 to 1)


stopFlag = False
detectRedFlag =False
red_detected_event = threading.Event()
green_detected_event = threading.Event()
blue_detected_event = threading.Event()
yellow_detected_event = threading.Event()
purple_detected_event = threading.Event()

detectGreenFlag =False
detectBlueFlag =False
detectPurpleFlag = False


def detect_color_thread():
    global detectRedFlag, detectGreenFlag, detectBlueFlag, detectPurpleFlag
    while True:
        imageFrame = picam2.capture_array()
        colors = detectColourNum(imageFrame)
        if "red" in colors:
            detectRedFlag = 1
            red_detected_event.set()
            #print("red detected")
        elif "green" in colors:
            detectGreenFlag = True
            green_detected_event.set()
            #print("green detected")
        elif "blue" in colors:
            detectBlueFlag = True
            blue_detected_event.set()
            #print("blue detected")
        elif "purple" in colors:
            detectPurpleFlag = True
            purple_detected_event.set()
            #print("purple detected")
        print(colors)
        colors.clear()

def detect_symbol_thread():
    global detectRedFlag, detectGreenFlag, detectBlueFlag, detectPurpleFlag, stopFlag
    while 1:
        if red_detected_event.is_set():
            stopFlag = True
            stop()
            print("test: detected red color")
            imageFrame = picam2.capture_array()
            className = detectRedSymbol(imageFrame)
            print(className)
            if className == "Line" or className == "Red Line":
                print("test: enter line")
                if "red" in preferred_colors:
                    print("for loop color line following start")
                    for i in range(1000):
                        imageFrame = picam2.capture_array()
                        colorLineFollow(imageFrame)
                    print("for loop color line following stop")

                    print("for loop overcoming color line start")
                    for i in range(50):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop overcoming color line stop")
                else:

                    print("for loop black line follow start")
                    for i in range(100):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop black line follow stop")
            else:
                print("Symbol Detected: ", className)
                sleep(2)
                print("for loop overcoming symbol start")
                for i in range(200):
                    imageFrame = picam2.capture_array()
                    lineFollow(imageFrame, look_ahead_cx=None)
                print("for loop overcoming symbol stop")
            red_detected_event.clear()
            detectRedFlag = False
            stopFlag = False
        elif green_detected_event.is_set():
            stopFlag = True
            stop()
            print("test: detected green color")
            imageFrame = picam2.capture_array()
            className = detectGreenSymbol(imageFrame)
            if className == "Line" or className == "Green Line":
                if "green" in preferred_colors:
                    print("for loop color line following start")
                    for i in range(1000):
                        imageFrame = picam2.capture_array()
                        colorLineFollow(imageFrame)
                    print("for loop color line following stop")

                    print("for loop overcoming color line start")
                    for i in range(50):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop overcoming color line stop")
                else:

                    print("for loop black line follow start")
                    for i in range(100):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop black line follow stop")
            else:
                print("Symbol Detected: ", className)
                sleep(2)
                print("for loop overcoming symbol start")
                for i in range(200):
                    imageFrame = picam2.capture_array()
                    lineFollow(imageFrame, look_ahead_cx=None)
                print("for loop overcoming symbol stop")
            green_detected_event.clear()
            detectGreenFlag = False
            stopFlag = False
        elif blue_detected_event.is_set():
            stopFlag = True
            stop()
            print("test: detected blue color")
            imageFrame = picam2.capture_array()
            className = detectBlueSymbol(imageFrame)
            if className == "Line" or className == "Blue Line":
                if "blue" in preferred_colors:
                    print("for loop color line following start")
                    for i in range(1000):
                        imageFrame = picam2.capture_array()
                        colorLineFollow(imageFrame)
                    print("for loop color line following stop")

                    print("for loop overcoming color line start")
                    for i in range(50):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop overcoming color line stop")
                else:

                    print("for loop black line follow start")
                    for i in range(100):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop black line follow stop")
            else:
                print("Symbol Detected: ", className)
                sleep(2)
                print("for loop overcoming symbol start")
                for i in range(200):
                    imageFrame = picam2.capture_array()
                    lineFollow(imageFrame, look_ahead_cx=None)
                print("for loop overcoming symbol stop")
            blue_detected_event.clear()
            detectBlueFlag = False
            stopFlag = False
        elif purple_detected_event.is_set():
            stopFlag = True
            stop()
            print("test: detected purple color")
            imageFrame = picam2.capture_array()
            className = detectPurpleSymbol(imageFrame)
            if className == "Line" or className == "Purple Line":
                if "purple" in preferred_colors:
                    print("for loop color line following start")
                    for i in range(1000):
                        imageFrame = picam2.capture_array()
                        colorLineFollow(imageFrame)
                    print("for loop color line following stop")

                    print("for loop overcoming color line start")
                    for i in range(50):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop overcoming color line stop")
                else:

                    print("for loop black line follow start")
                    for i in range(100):
                        imageFrame = picam2.capture_array()
                        lineFollow(imageFrame, look_ahead_cx=None)
                    print("for loop black line follow stop")
            else:
                print("Symbol Detected: ", className)
                sleep(2)
                print("for loop overcoming symbol start")
                for i in range(200):
                    imageFrame = picam2.capture_array()
                    lineFollow(imageFrame, look_ahead_cx=None)
                print("for loop overcoming symbol stop")
            detectPurpleFlag = False
            purple_detected_event.clear()
            stopFlag = False


def line_following_thread():
    global stopFlag, redKnowledgeFlag
    while 1:
        if stopFlag is False:
            # print("I want to line follow!!!")
            imageFrame = picam2.capture_array()
            lineFollow(imageFrame, look_ahead_cx=None)


def detectRedSymbol(crop_img):  # a function to determine the class name
    #resolution224()
    

    image_resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1

    # Predict the model
    prediction = modelRed.predict(image_for_prediction)
    index = np.argmax(prediction)
    class_name = class_namesRed[index]
    # confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    num_str = str(index)  # change index type to str
    length_str = len(num_str)  # Get the length of the string
    initial = length_str + 1  # get initial
    #resolution192()
    return class_name[initial:].strip()


def detectGreenSymbol(crop_img):  # a function to determine the class name
    #resolution224()
    

    image_resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1

    # Predict the model
    prediction = modelGreen.predict(image_for_prediction)
    index = np.argmax(prediction)
    class_name = class_namesGreen[index]
    # confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    num_str = str(index)  # change index type to str
    length_str = len(num_str)  # Get the length of the string
    initial = length_str + 1  # get initial
    #resolution192()
    return class_name[initial:].strip()


def detectBlueSymbol(crop_img):  # a function to determine the class name
    #resolution224()

    image_resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1

    # Predict the model
    prediction = modelBlue.predict(image_for_prediction)
    index = np.argmax(prediction)
    class_name = class_namesBlue[index]
    # confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    num_str = str(index)  # change index type to str
    length_str = len(num_str)  # Get the length of the string
    initial = length_str + 1  # get initial
    #resolution192()
    return class_name[initial:].strip()


def detectPurpleSymbol(crop_img):  # a function to determine the class name
    #resolution224()

    image_resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1

    # Predict the model
    prediction = modelPurple.predict(image_for_prediction)
    index = np.argmax(prediction)
    class_name = class_namesPurple[index]
    # confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    num_str = str(index)  # change index type to str
    length_str = len(num_str)  # Get the length of the string
    initial = length_str + 1  # get initial
    #resolution192()
    return class_name[initial:].strip()


def forward():
    in1.on()  # blue dc motor
    in2.off()
    in3.on()  # white dc motor
    in4.off()
    # en1.value = 0.34  # Set motor speed (0 to 1)
    # en2.value = 0.34  # Set motor speed (0 to 1)


def stop():
    in1.on()  # blue dc motor
    in2.on()
    in3.on()  # white dc motor
    in4.on()
    en1.value = 0.0  # Set motor speed (0 to 1)
    en2.value = 0.0  # Set motor speed (0 to 1)
    # print("I don't see the line")


def backwardRight():
    in1.off()  # blue dc motor
    in2.on()
    in3.on()  # white dc motor
    in4.off()
    # en1.value = 0.5  # Set motor speed (0 to 1)
    # en2.value = 0.5  # Set motor speed (0 to 1)


def backwardLeft():
    in1.on()  # blue dc motor
    in2.off()
    in3.off()  # white dc motor
    in4.on()
    # en1.value = 0.5  # Set motor speed (0 to 1)
    # en2.value = 0.5  # Set motor speed (0 to 1)


def right():
    in1.on()  # blue dc motor
    in2.off()
    in3.off()  # white dc motor
    in4.on()
    # en1.value =0.5  # Set motor speed (0 to 1)
    # en2.value = 0.5  # Set motor speed (0 to 1)


def left():
    in1.off()  # blue dc motor
    in2.on()
    in3.on()  # white dc motor
    in4.off()
    # en1.value = 0.53  # Set motor speed (0 to 1)
    # en2.value = 0.53  # Set motor speed (0 to 1)


def get_look_ahead_position(frame_count, look_ahead_cx=None):
    global prev_look_ahead_cx

    if frame_count < look_ahead_frames:
        frame_count += 1
        crop_img = picam2.capture_array()
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Color thresholding
        ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(thresh, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                return cx, frame_count

    if prev_look_ahead_cx is not None:
        return prev_look_ahead_cx, frame_count

    return None, frame_count


prev_look_ahead_cx = None


def lineFollow(crop_img, look_ahead_cx=None):
    global prev_error, integral, derivative, prev_look_ahead_cx

    crop_img = picam2.capture_array()
    rgbFrame = crop_img[:, :, ::-1]
    black_lower = np.array([10, 20, 20], np.uint8)
    black_upper = np.array([50, 60, 50], np.uint8)
    mask = cv2.inRange(rgbFrame, black_lower, black_upper)
    # gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Color thresholding
    # ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    # mask = cv2.erode(thresh, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)

    contours, hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)

    # Find the biggest contour (if detected)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])

            error = setpoint - cx
            integral += error
            derivative = error - prev_error
            prev_error = error

            # Calculate the PID output
            output = Kp * error + Ki * integral + Kd * derivative

            # Get look-ahead position
            look_ahead_cx, frame_count = get_look_ahead_position(0)

            if look_ahead_cx is not None:
                look_ahead_error = look_ahead_setpoint - look_ahead_cx

                # Adjust the output based on the look-ahead error
                output += Kp * look_ahead_error * 0.1

            if cx <= 40:
                print("Turn Left!")
                left()
                en1.value = max(0, min(0.38 - output, 1))  # Set motor speed (0 to 1)
                en2.value = max(0, min(0.38 + output, 1))  # Set motor speed (0 to 1)
            elif cx >= 150:
                print("Turn Right")
                right()
                en1.value = max(0, min(0.38 - output, 1))  # Set motor speed (0 to 1)
                en2.value = max(0, min(0.38 + output, 1))  # Set motor speed (0 to 1)

            else:
                print("On Track!")
                # print("im here")
                forward()
                en1.value = max(0, min(0.33 - output, 1))  # Set motor speed (0 to 1)
                en2.value = max(0, min(0.33 + output, 1))  # Set motor speed (0 to 1)
                # en1.value = max(0, min(0.45 - output, 1))  # Set motor speed (0 to 1)
                # en2.value = max(0, min(0.45 + output, 1))  # Set motor speed (0 to 1)

    else:
        print("No contours found, initiating search...")
        stop()
        # sleep(0.5)
        backward()
        en1.value = 0.5  # Set motor speed (0 to 1)
        en2.value = 0.5  # Set motor speed (0 to 1)
        sleep(0.1)
    cv2.imshow("black line", mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    prev_look_ahead_cx = look_ahead_cx


def colorLineFollow(crop_img):
    # rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    rgb = crop_img[:, :, ::-1]

    # Define color ranges based on user preference
    if "yellow" in preferred_colors:
        mask = cv2.inRange(rgb, yellow_lower, yellow_upper)

    elif "red" in preferred_colors:
        mask = cv2.inRange(rgb, red_lower, red_upper)

    elif "green" in preferred_colors:
        mask = cv2.inRange(rgb, green_lower, green_upper)

        # color_upper = np.array([80, 255, 80])

    elif "blue" in preferred_colors:
        mask = cv2.inRange(rgb, blue_lower, blue_upper)

    # Mask for black and preferred color

    detected_color = cv2.bitwise_and(rgb, rgb, mask=mask)
    ##########black_mask = cv2.inRange(rgb, black_lower, black_upper)
    # color_mask = cv2.inRange(rgb, color_lower, color_upper)
    # mask = cv2.bitwise_or(black_mask, color_mask)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
        else:
            cx = 0

        if cx <= 40:
            print("Colour Turn Left!")
            in1.off()  # blue dc motor
            in2.on()
            in3.on()  # white dc motor
            in4.off()
            en1.value = 0.65  # Set motor speed (0 to 1)
            en2.value = 0.65  # Set motor speed (0 to 1)
            # time.sleep(0.1)
        elif 40 < cx < 150:
            print("On Coloured Track!")
            in1.on()  # blue dc motor
            in2.off()
            in3.on()  # white dc motor
            in4.off()
            en1.value = 0.31  # Set motor speed (0 to 1)
            en2.value = 0.31  # Set motor speed (0 to 1)
        else:
            print("Colour Turn Right")
            in1.on()  # blue dc motor
            in2.off()
            in3.off()  # white dc motor
            in4.on()
            en1.value = 0.65  # Set motor speed (0 to 1)
            en2.value = 0.65  # Set motor speed (0 to 1)
            # time.sleep(0.1)
    else:
        print("Colour I dont see the line")
        imageFrame = picam2.capture_array()
        lineFollow(imageFrame)
        # backward()
        """"
        in1.on()  # blue dc motor
        in2.on()
        in3.on()  # white dc motor
        in4.on()
        en1.value = 0.0  # Set motor speed (0 to 1)
        en2.value = 0.0  # Set motor speed (0 to 1)
        print("I don't see the line")
        in1.on()  # blue dc motor
        in2.off()
        in3.off()  # white dc motor
        in4.on()
        en1.value = 0.6  # Set motor speed (0 to 1)
        en2.value = 0.6  # Set motor speed (0 to 1)
        sleep(0.1)
        """

    cv2.imshow("babi colour ", mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def resolution224():
    print("Resolution set to 224x224!")
    # picam2 = Picamera2()
    picam2.stop()
    picam2.resolution = (224, 224)
    # picam2.framerate = 35
    picam2.preview_configuration.main.size = (224, 224)
    # picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    sleep(2)
    
def resolution192():
    print("Resolution set to 224x224!")
    # picam2 = Picamera2()
    picam2.stop()
    picam2.resolution = (192,108)
    # picam2.framerate = 35
    picam2.preview_configuration.main.size = (192,108)
    # picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    sleep(2)
    
def detectColourNum(imageFrame):
    rgbFrame = imageFrame[:, :, ::-1]
    # rgbFrame = cv2.cvtColor(imageFrame, cv2.COLOR_rgb2BGR)

    # Define RGB ranges for color detection
    red_mask = cv2.inRange(rgbFrame, red_lower, red_upper)
    # res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    green_mask = cv2.inRange(rgbFrame, green_lower, green_upper)
    # res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

    blue_mask = cv2.inRange(rgbFrame, blue_lower, blue_upper)
    # res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    black_mask = cv2.inRange(rgbFrame, black_lower, black_upper)
    # res_black = cv2.bitwise_and(imageFrame, imageFrame, mask=black_mask)

    yellow_mask = cv2.inRange(rgbFrame, yellow_lower, yellow_upper)
    # res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)

    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            # x, y, w, h = cv2.boundingRect(contour)
            # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            detected_colors.add("red")

    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            # x, y, w, h = cv2.boundingRect(contour)
            # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.putText(imageFrame, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            detected_colors.add("yellow")

    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            # x, y, w, h = cv2.boundingRect(contour)
            # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            detected_colors.add("green")

    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            # x, y, w, h = cv2.boundingRect(contour)
            # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
            detected_colors.add("blue")

    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            # x, y, w, h = cv2.boundingRect(contour)
            # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (100, 255, 0), 2)
            # cv2.putText(imageFrame, "Black Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 0))
            detected_colors.add("black")
    return detected_colors


try:
    while 1:
        color_thread = threading.Thread(target=detect_color_thread)

        # colorLine_thread = threading.Thread(target=colorLine_following_thread)
        symbol_thread = threading.Thread(target=detect_symbol_thread)
        line_thread = threading.Thread(target=line_following_thread)

        color_thread.start()
        line_thread.start()
        # colorLine_thread.start()
        symbol_thread.start()
        
        ####
        color_thread.join()  # Wait for the color detection thread to finish
        line_thread.join()  # Wait for the line following thread to finish
        # colorLine_thread.join()
        symbol_thread.join()

        detected_colors.clear()
        print("All Thread finish !")
        color_detected_event.clear()
        symbol_detected_event.clear()

    stop()
    in1.close()
    in2.close()
    in3.close()
    in4.close()
    en1.close()
    en2.close()
except KeyboardInterrupt:
    stop()
    in1.close()
    in2.close()
    in3.close()
    in4.close()
    en1.close()
    en2.close()
    print("Keyboard interrupt detected. Stopping motors.")

finally:
    # Cleanup GPIO pins
    in1.close()
    in2.close()
    in3.close()
    in4.close()
    en1.close()
    en2.close()
    print("Motors stopped. Program terminated.")






