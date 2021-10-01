import numpy as np
import cv2 as cv

from TP1.contour import get_contours, get_biggest_contour, compare_contours
from TP1.trackbar import create_trackbar, get_trackbar_value, adaptive_threshold
from color_map import normal, binary

saved_contours = []
alpha_slider_max = 255

window_name = 'Grey'
trackbar_name = 'Trackbar'
cv.namedWindow(window_name)
create_trackbar(trackbar_name, window_name, alpha_slider_max)

window_name2 = 'sure_bg'
trackbar_name2 = 'Trackbar_Bg'
cv.namedWindow(window_name2)
create_trackbar(trackbar_name2, window_name2, alpha_slider_max)

while True:
    img = cv.imread('./image/levadura.png')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    trackbar_val = get_trackbar_value(trackbar_name, window_name)
    _, thresh = cv.threshold(grey, trackbar_val, 255, cv.THRESH_BINARY)
    #+cv.THRESH_OTSU
    cv.imshow(window_name, thresh)

    trackbar_val2 = get_trackbar_value(trackbar_name2, window_name2)
    l, thresh2 = cv.threshold(grey, trackbar_val2, 255, cv.THRESH_BINARY)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)

    # Finding sure foreground area
    sure_fg = cv.erode(closing, kernel, iterations=3)
    # sure background area
    opening2 = cv.morphologyEx(thresh2, cv.MORPH_OPEN, kernel, iterations=2)
    closing2 = cv.morphologyEx(opening2, cv.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv.dilate(closing2, kernel, iterations=3)

    contours, h = cv.findContours(sure_fg, 1, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv.drawContours(sure_fg, [cnt], 0, (0, 0, 255), 2)#antes estaba img



    img = cv.putText(img, str(len(contours)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                  2, cv.LINE_AA)
    cv.imshow("sure_fg", sure_fg)
    cv.imshow("img", img)
    cv.imshow("sure_bg", sure_bg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)

    unknown = cv.subtract(sure_bg, sure_fg)
    cv.imshow("Unknown", unknown)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)

    img[markers == -1] = [255, 0, 0]





    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.waitKey()
        cv.imshow("watershed", img)
        normal(sure_bg)
        #binary(img,sure_bg,thresh2)
        #binary(img,sure_fg,thresh)

        break






