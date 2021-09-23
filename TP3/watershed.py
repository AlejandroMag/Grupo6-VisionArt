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



    contours, h = cv.findContours(sure_bg, 1, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv.drawContours(img, [cnt], 0, (0, 0, 255), 2)

    cv.imshow("sure_bg", img)

    img = cv.putText(img, str(len(contours)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                  2, cv.LINE_AA)
    cv.imshow("sure_bg", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#cv.imshow("img", thresh)

cv.waitKey()



#cv.imshow("sure_bg", sure_bg)

cv.waitKey()