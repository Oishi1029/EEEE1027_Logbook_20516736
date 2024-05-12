import numpy as np
import cv2
#import time
from picamera2 import Picamera2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice
from time import sleep
from keras.models import load_model
from gtts import gTTS
import random
preferred_color = input("Enter the preferred color (yellow, red, green, blue): ").lower()
red_lower = np.array([65, 0, 5], np.uint8)
red_upper = np.array([210, 45, 50], np.uint8)

#green_lower = np.array([5, 130, 75], np.uint8)
#green_upper = np.array([20, 141, 90], np.uint8)

green_lower = np.array([0, 80, 37], np.uint8)
green_upper = np.array([5, 170, 92], np.uint8)

blue_lower = np.array([7, 16, 48], np.uint8)
blue_upper = np.array([22, 33, 100], np.uint8)

# Add black color detection
black_lower = np.array([15, 17, 15], np.uint8)
black_upper = np.array([39, 40, 37], np.uint8)

yellow_lower = np.array([205, 170, 5], np.uint8)
yellow_upper = np.array([230, 210, 50], np.uint8)

kernel = np.ones((5, 5), "uint8")

# PID controller parameters
Kp = 0.0080
Ki = 0.0
Kd = 0.0

setpoint = 102.5  # Desired distance from the line
# setpoint = 80  # Desired distance from the line
prev_error = 0
integral = 0
derivative = 0

# Define GPIO pins
in1_pin = 24
in2_pin = 23
in3_pin = 27
in4_pin = 22
en1_pin = 12
en2_pin = 13

# Initialize GPIO devices
in1 = DigitalOutputDevice(in1_pin)
in2 = DigitalOutputDevice(in2_pin)
in3 = DigitalOutputDevice(in3_pin)
in4 = DigitalOutputDevice(in4_pin)
en1 = PWMOutputDevice(en1_pin)
en2 = PWMOutputDevice(en2_pin)

# start camera
picam2 = Picamera2()
picam2.resolution = (192, 108)
picam2.framerate = 30
picam2.preview_configuration.main.size = (192, 108)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
sleep(2)

# Initialize color sets
detected_colors = set()

# load model and classname
model = load_model("/home/dean/Documents/testForPW3Combined/keras_allSymbol/keras_model.h5", compile=False)
class_names = open("/home/dean/Documents/testForPW3Combined/keras_allSymbol/labels.txt", "r").readlines()


def detectSymbol():  # a function to determine the class name
    crop_img = picam2.capture_array()
    image_resized = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1

    # Predict the model
    prediction = model.predict(image_for_prediction)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    num_str = str(index)  # change index type to str
    length_str = len(num_str)  # Get the length of the string
    initial = length_str + 1  # get initial

    return class_name[initial:].strip()
    # Draw bounding box
    # readSymbol(class_name.strip(), index, crop_img)


def backward():
    in1.off()  # blue dc motor
    in2.on()
    in3.off()  # white dc motor
    in4.on()
    en1.value = 0.34  # Set motor speed (0 to 1)
    en2.value = 0.34  # Set motor speed (0 to 1)


def forward():
    in1.on()  # blue dc motor
    in2.off()
    in3.on()  # white dc motor
    in4.off()
    en1.value = 0.34  # Set motor speed (0 to 1)
    en2.value = 0.34  # Set motor speed (0 to 1)


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
    en1.value = 0.5  # Set motor speed (0 to 1)
    en2.value = 0.5  # Set motor speed (0 to 1)


def backwardLeft():
    in1.on()  # blue dc motor
    in2.off()
    in3.off()  # white dc motor
    in4.on()
    en1.value = 0.5  # Set motor speed (0 to 1)
    en2.value = 0.5  # Set motor speed (0 to 1)


def right():
    in1.on()  # blue dc motor
    in2.off()
    in3.off()  # white dc motor
    in4.on()
    en1.value = 0.55  # Set motor speed (0 to 1)
    en2.value = 0.55  # Set motor speed (0 to 1)


def left():
    in1.off()  # blue dc motor
    in2.on()
    in3.on()  # white dc motor
    in4.off()
    en1.value = 0.6  # Set motor speed (0 to 1)
    en2.value = 0.6  # Set motor speed (0 to 1)



def detectRGB():
    imageFrame = picam2.capture_array()


    kernel = np.ones((5, 5), "uint8")

    rgbFrame = imageFrame[:, :, ::-1]

    # Define RGB ranges for color detection
    color_ranges = {
        'red': (red_lower, red_upper),
        'green': (green_lower, green_upper),
        'blue': (blue_lower, blue_upper),
        'black': (black_lower, black_upper),
        'yellow': (yellow_lower, yellow_upper)
    }

    detected_colors = set()

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(rgbFrame, lower, upper)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                detected_colors.add(color)
                #break
    print("colors detected:", detected_colors)

    return detected_colors

def detectHSV(imageFrame):
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Add black color detection
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 70], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    kernel = np.ones((5, 5), "uint8")

    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    black_mask = cv2.dilate(black_mask, kernel)
    res_black = cv2.bitwise_and(imageFrame, imageFrame, mask=black_mask)

    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            detected_colors.add("red")
            print("red detected")

    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            detected_colors.add("green")
            print("green detected")

    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
            detected_colors.add("blue")
            print("blue detected")

    contours, hierarchy = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (100, 255, 0), 2)
            cv2.putText(imageFrame, "Black Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 0))
            detected_colors.add("black")
            print("black detected")

    return detected_colors

def getCountoursLen():
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
        else:
            cx = 0
    else:
        cx = -1
    return len(contours), cx

def lineFollow(look_ahead_cx=None):
    global prev_error, integral, derivative, prev_look_ahead_cx, backwardCounter

    crop_img = picam2.capture_array()
    rgbFrame = crop_img[:, :, ::-1]
    black_lower = np.array([10, 20, 20], np.uint8)
    black_upper = np.array([50, 60, 50], np.uint8)

    # gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Color thresholding
    # ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    # mask = cv2.erode(thresh, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.inRange(rgbFrame, black_lower, black_upper)
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
                forward()
                en1.value = max(0, min(0.35 - output, 1))  # Set motor speed (0 to 1)
                en2.value = max(0, min(0.35 + output, 1))  # Set motor speed (0 to 1)

    else:
        print("No contours found, initiating search...")
        stop()
        # sleep(0.5)
        backward()
        en1.value = 0.5  # Set motor speed (0 to 1)
        en2.value = 0.5  # Set motor speed (0 to 1)
        sleep(0.1)
        backwardCounter = backwardCounter + 1
        if backwardCounter == 5:
            right()
            sleep(1)
            backwardCounter = 0

        """"
        backwardCounter = backwardCounter + 1
        #testFlag = 1
        if backwardCounter == 7:
            for i in range(200):
                highSpeedLineFollow()
        """
    prev_look_ahead_cx = look_ahead_cx


def hardCodeRight():
    right()
    random_float = random.random()
    # print("Random Float:", random_float)
    sleep(0.09)
    # sleep(random_float/5)


def hardCodeLeft():
    left()
    random_float = random.random()
    # print("Random Float:", random_float)
    sleep(random_float / 5)


def hardCodeBack():
    backward()
    random_float = random.random()
    # print("Random Float:", random_float)
    sleep(random_float / 15)


def hardCodeBackRight():
    backwardRight()
    random_float = random.random()
    # print("Random Float:", random_float)
    sleep(random_float / 15)


def hardCodeBackLeft():
    backwardLeft()
    random_float = random.random()
    # print("Random Float:", random_float)
    sleep(random_float / 15)


def colorLineFollow():
    crop_img = picam2.capture_array()
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

    # Define color ranges based on user preference
    if preferred_color == "yellow":
        mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    elif preferred_color == "red":
        mask = cv2.inRange(hsv, red_lower, red_upper)

    elif preferred_color == "green":
        mask = cv2.inRange(hsv, green_lower, green_upper)


    elif preferred_color == "blue":
        mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Mask for black and preferred color

    #detected_color = cv2.bitwise_and(hsv, hsv, mask=mask)
    ##########black_mask = cv2.inRange(hsv, black_lower, black_upper)
    # color_mask = cv2.inRange(hsv, color_lower, color_upper)
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
            print("Turn Left!")
            in1.off()  # blue dc motor
            in2.on()
            in3.on()  # white dc motor
            in4.off()
            en1.value = 0.65  # Set motor speed (0 to 1)
            en2.value = 0.65  # Set motor speed (0 to 1)
            time.sleep(0.1)
        elif 40 < cx < 150:
            print("On Track!")
            in1.on()  # blue dc motor
            in2.off()
            in3.on()  # white dc motor
            in4.off()
            en1.value = 0.31  # Set motor speed (0 to 1)
            en2.value = 0.31  # Set motor speed (0 to 1)
        else:
            print("Turn Right")
            in1.on()  # blue dc motor
            in2.off()
            in3.off()  # white dc motor
            in4.on()
            en1.value = 0.65  # Set motor speed (0 to 1)
            en2.value = 0.65  # Set motor speed (0 to 1)
            time.sleep(0.1)
    else:
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
        sleep(0.35)

    cv2.imshow("babi colour ", mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def executeTask(name):
    # num_str = str(index)  # change index type to str
    # length_str = len(num_str)  # Get the length of the string
    # initial = length_str + 1  # get initial
    img = picam2.capture_array()
    lineList = ["Line", "Background", "Blue Line", "Red Line","Green Line"]
    if name in lineList:
        detected_colors = detectHSV(img)
        #detected_colors = detectRGB(img)
        if preferred_color in detected_colors:
            colorLineFollow(preferred_color)

        else:  # normal line follow otherwise
            lineFollow()
    else:

        resolution224()
        imageFrame = picam2.capture_array()
        name = detectSymbol(imageFrame)  # obtain class name and index
        print("Symbol detected:", name)
        print("stop for 2 seconds")
        sleep(2)
        print("OVERCOME SYMBOL")
        resolution192()
        # picam2.resolution(192,108)
        for i in range(50):
            overcomeSymbol()


def overcomeSymbol():
    crop_img = picam2.capture_array()
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Color thresholding
    ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.erode(thresh, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
    # Find the contours of the frame
    # contours,hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)
    # Find the biggest contour (if detected)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])

        else:
            cx = 0

        if cx <= 40:
            print("Overcome Turn Left!")
            left()

        if cx < 150 and cx > 40:
            print("Overcome On Track!")
            forward()

        if cx >= 150:
            print("Overcome Turn Right")
            right()
    else:
        print("overcome no contour found")
        # stop()
        # right()
        # sleep(0.05)
        hardCodeRight()
        # hardCode()


def resolution192():
    picam2.stop()
    picam2.resolution = (192, 108)
    # picam2.framerate = 35
    picam2.preview_configuration.main.size = (192, 108)
    # picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    sleep(2)


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


def playSound(name):
    myobj = gTTS(text=name[initial:], lang=language, slow=False)
    myobj.save("welcome.mp3")

    os.system("mpg321 welcome.mp3")  # play sound

def findContours():
    imageFrame = picam2.capture_array()
    img = cv2.medianBlur(imageFrame, 5)  # not necessary to have
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # adjust value based on the gray frame

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return contours
try:
    while (1):
        contours = findContours()
        if len(contours) > 2:
            print("Found {} contours".format(len(contours)))
            stop()

            name = detectSymbol()  # obtain class name and index

            print(f"Object: {name}")
            lineList = ["Line", "Background", "Blue Line", "Red Line", "Green Line"]

            if name in lineList:
                #detected_colors = detectHSV(img)
                detected_colors = detectRGB()
                if preferred_color in detected_colors:
                    colorLineFollow()

                else:  # normal line follow otherwise
                    lineFollow()
            else:
                resolution224()
                name = detectSymbol()  # obtain class name and index
                print("Symbol detected:", name)
                print("stop for 2 seconds")
                sleep(2)
                print("OVERCOME SYMBOL")
                resolution192()
                # picam2.resolution(192,108)
                for i in range(50):
                    overcomeSymbol()
        else:
            colors = detectRGB()
            #colors = detectHSV()
            if "black" in colors:
                lineFollow()
            else:
                colorLineFollow()
        detected_colors.clear()

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Stopping motors.")
    stop()
finally:
    # Cleanup GPIO pins
    stop()
    in1.close()
    in2.close()
    in3.close()
    in4.close()
    en1.close()
    en2.close()
    print("Motors stopped. Program terminated.")
