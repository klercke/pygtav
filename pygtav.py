import numpy as np
from PIL import ImageGrab
import cv2
import time
from directxkeys import *


def roi(img, verticies):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, verticies, 255)
    masked = cv2.bitwise_and(img, mask)

    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    verticies = np.array([
        [10, 500], 
        [10, 300], 
        [300, 200], 
        [500, 200], 
        [800, 300], 
        [800, 500]
    ])
    processed_img = roi(processed_img, [verticies])

    return processed_img


def main():
    """ for i in list(range(3)) [::-1]:
        print(i+1)
        time.sleep(1) """


    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox = (0, 40, 800, 640)))
        new_screen = process_img(screen)
        
        print("Loop took {} seconds".format(time.time() - last_time))
        last_time = time.time()

        # cv2.imshow('PyGTAV (Unprocessed)', cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
        cv2.imshow('PyGTAV', new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        

if __name__ == "__main__":
    main()