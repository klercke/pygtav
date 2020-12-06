import numpy as np
from PIL import ImageGrab
import cv2
import time
from directxkeys import *


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
    except:
        pass


def roi(img, verticies):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, verticies, 255)
    masked = cv2.bitwise_and(img, mask)

    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)

    verticies = np.array([
        [10, 500], 
        [10, 300], 
        [300, 200], 
        [500, 200], 
        [800, 300], 
        [800, 500]
    ])
    processed_img = roi(processed_img, [verticies])

    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, None, 30, 15)
    draw_lines(processed_img, lines)

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