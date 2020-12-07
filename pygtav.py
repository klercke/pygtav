import numpy as np
from PIL import ImageGrab
import cv2
import time
from directxkeys import *
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
import traceback

# Difference in slope of lines that can be considerend a single line
LANE_SIMILARITY = .20

# Minimum pixel length for lines to be detected
MIN_LINE_LENGTH = 150

# Maximum pixel gap for lines to be considered cohesive
MAX_GAP_SIZE = 15

# Lane line width and color
DETECTED_LANE_COLOR = [0, 255, 0]
DETECTED_LANE_WIDTH = 30


def draw_lanes(img, lines):
    # try to draw lines intelligently
    try:
        # find maximum y value for lane marker (horizon)
        ys = []
        for i in lines:
            for j in i:
                ys += [j[1], j[3]]
        
        min_y = min(ys)
        max_y = 600
        # new_lines = []
        line_dict = {}

        for idx, i in enumerate(lines):
            for xyxy in i:
                # https://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Calculate line qualities (slope, coords, etc.) based on two sets of coords
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]
                # Calculate improved x values
                x1 = (min_y - b) / m
                x2 = (max_y - b) / m

                line_dict[idx] = [m, b, [int(x1), min_y, int(x2), max_y]]
                # new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [[m, b, line]]

            else:
                found_copy = False
                
                for other_ms in final_lanes_copy:
                    if not found_copy:
                        # detect similar slopes (+/- LANE_SIMILARITY of an already detected line) and ignore them
                        if abs(other_ms * (1 + LANE_SIMILARITY)) > abs(m) > abs(other_ms * (1 - LANE_SIMILARITY)):
                            if abs(final_lanes_copy[other_ms][0][1] * (1 + LANE_SIMILARITY)) > abs(b) > abs(final_lanes_copy[other_ms][0][1] * (1 - LANE_SIMILARITY)):
                                final_lanes[other_ms].append([m, b, line])
                                found_copy = True
                                break

                        else:
                            final_lanes[m] = [[m, b, line]]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key = lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [[l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]]
    
    except Exception as e:
        print(traceback.format_exc())




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

    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, None, MIN_LINE_LENGTH, MAX_GAP_SIZE)
    # draw_lines(processed_img, lines)

    try:
        l1, l2 = draw_lanes(original_image, lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), DETECTED_LANE_COLOR, DETECTED_LANE_WIDTH)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), DETECTED_LANE_COLOR, DETECTED_LANE_WIDTH)

    except Exception as e:
        print(traceback.format_exc())
        pass

    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)

            except Exception as e:
                print(traceback.format_exc())

    except Exception as e:
        pass

    return processed_img, original_image


def main():
    """ for i in list(range(3)) [::-1]:
        print(i+1)
        time.sleep(1) """


    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox = (0, 40, 800, 640)))
        print("Frame took {} seconds".format(time.time() - last_time))
        last_time = time.time()

        new_screen, original_image = process_img(screen)

        cv2.imshow('PyGTAV (Processed)', new_screen)
        cv2.imshow('PyGTAV (Lanes)', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        

if __name__ == "__main__":
    main()