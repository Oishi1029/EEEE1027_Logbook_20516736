import time 
import numpy as np 
from picamera2 import Picamera2 

# Picamera2 setup\ 

h = 640  # Change this to anything < 2592 (anything over 2000 will likely get a memory error when plotting)\ 
cam_res = (int(h), int(0.75 * h))  # Keeping the natural 3/4 resolution of the camera\ 
cam_res = (int(16 * np.floor(cam_res[1] / 16)), int(32 * np.floor(cam_res[0] / 32))) 

picam2 = Picamera2() 
config = picam2.create_preview_configuration(main={"size": (cam_res[1], cam_res[0])}) 
picam2.configure(config) 
picam2.start() 

# Make sure the picamera2 doesn't change white balance or exposure\ 
# This will help create consistent images\ 
time.sleep(2)  # Let the camera settle\ 

# The picamera2 library handles ISO, shutter speed, and other settings differently,\ 
# so this part of the code is omitted. You can use picam2's manual controls if needed.\ 

# Prepping for analysis and recording background noise\ 

# The objects should be removed while background noise is calibrated\ 
noise = picam2.capture_array() 
# Subtract the mean to get the background noise\ 
noise = noise - np.mean(noise) 
# Looping with different images to determine instantaneous colors\ 
rgb_text = ['Red', 'Green', 'Blue']  # Array for naming color\ 
input("Press enter to capture background noise (remove colors)") 
while True: 
    try: 
        print('===========================') 
        input("Press enter to capture image") 
        data = picam2.capture_array() 
        mean_array, std_array = [], [] 
        for ii in range(0, 3): 
            # Calculate mean and STDev and print out for each color\ 
            corrected_data = data[:, :, ii] - np.mean(data) - noise[:, :, ii] 
            mean_array.append(np.mean(corrected_data)) 
            std_array.append(np.std(corrected_data)) 
            print('-------------------------') 
            print(f'{rgb_text[ii]}---mean: {mean_array[ii]:2.1f}, stdev: {std_array[ii]:2.1f}') 
        # Guess the color of the object\ 
        print('--------------------------') 
        print(f'The Object is: {rgb_text[np.argmax(mean_array)]}') 
        print('--------------------------') 
    except KeyboardInterrupt: 
        break 
