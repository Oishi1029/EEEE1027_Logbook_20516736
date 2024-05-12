import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (192,108)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
cpt = 0
maxFrames = 1000 #this means 30 image will be captured into a folder
while cpt < maxFrames:
    im= picam2.capture_array()
    #im=cv2.flip(im,-1)
    cv2.imshow("Camera", im)
    
    cpt += 1
    #time.sleep(1)
    if cpt%10==0:
        cv2.imwrite('/home/dean/Documents/PW3/Left image/arduino_uno_%d.jpg' %cpt, im)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
