import numpy as np
import cv2 as cv

from TP1.contour import get_contours, get_biggest_contour, compare_contours
from TP1.trackbar import create_trackbar, get_trackbar_value, adaptive_threshold

saved_contours = []
alpha_slider_max = 255
window_name = 'Grey'
trackbar_name = 'Trackbar'
cv.namedWindow(window_name)
create_trackbar(trackbar_name, window_name, alpha_slider_max)


while True:
    img = cv.imread('./image/levadura.png')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    trackbar_val = get_trackbar_value(trackbar_name, window_name)
    _, thresh = cv.threshold(grey, trackbar_val, 255, cv.THRESH_BINARY)
    #+cv.THRESH_OTSU
    cv.imshow(window_name, thresh)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(closing, kernel, iterations=3)

    # Finding sure foreground area
    sure_fg = cv.erode(closing, kernel, iterations=3)

    contours, h = cv.findContours(sure_bg, 1, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    cv.imshow("sure_bg", img)

    img = cv.putText(img, str(len(contours)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                  2, cv.LINE_AA)
    cv.imshow("sure_bg", img)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)
    print(markers)
    print("------------------")
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)

    print(img)
    print(markers)

    img[markers == -1] = [255, 0, 0]



    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.imshow("img", img)
        cv.waitKey()
        break
