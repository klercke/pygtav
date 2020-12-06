import numpy as np
from PIL import ImageGrab
import cv2
import time
from directxkeys import *


for i in list(range(5)) [::-1]:
    print(i+1)
    time.sleep(1)

print('down')
PressKey(KEY_W)
time.sleep(3)
print('up')
ReleaseKey(KEY_W)


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    return processed_img


""" last_time = time.time()
while True:
    screen = np.array(ImageGrab.grab(bbox = (0, 40, 800, 640)))
    new_screen = process_img(screen)
    
    # print("Loop took {} seconds".format(time.time() - last_time))
    last_time = time.time()

    # cv2.imshow('PyGTAV (Unprocessed)', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
    cv2.imshow('PyGTAV', new_screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break """
    