from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from  glob import glob

def find_cells(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=5, maxRadius=100)
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 5)
        # draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    print("Total number of cells find in ", img_path,' is:',len(circles[0]))
    return circles[0]


if __name__=="__main__":
    img_path = "."
    for img in glob(img_path+"/*.png"):
        l_size = find_cells(img)
