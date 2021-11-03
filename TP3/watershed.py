import numpy as np
import cv2 as cv
from skimage import color
from Utiles.area import contour_area

from TP1.contour import get_contours, get_biggest_contour, compare_contours
from TP1.trackbar import create_trackbar, get_trackbar_value, adaptive_threshold
from color_map import normal, binary

saved_contours = []
alpha_slider_max = 255

window_name = 'FG'
trackbar_name = 'Trackbar'
cv.namedWindow(window_name)
create_trackbar(trackbar_name, window_name, alpha_slider_max)

window_name2 = 'BG'
trackbar_name2 = 'Trackbar_Bg'
cv.namedWindow(window_name2)
create_trackbar(trackbar_name2, window_name2, alpha_slider_max)

while True:
    img = cv.imread('./image/levadura.png')
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    trackbar_val = get_trackbar_value(trackbar_name, window_name)
    _, thresh = cv.threshold(grey, trackbar_val, 255, cv.THRESH_BINARY)

    trackbar_val2 = get_trackbar_value(trackbar_name2, window_name2)
    l, thresh2 = cv.threshold(grey, trackbar_val2, 255, cv.THRESH_BINARY)

    # +cv.THRESH_OTSU
    cv.imshow(window_name, thresh)
    cv.imshow(window_name2, thresh2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        kernel = np.ones((3, 3), np.uint8)

        # Finding sure foreground area sureFG
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
        closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

        sure_fg = cv.dilate(opening, kernel, iterations=4)

        # noise removal sureBG

        opening2 = cv.morphologyEx(thresh2, cv.MORPH_OPEN, kernel, iterations=1)
        closing2 = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, kernel, iterations=1)
        sure_bg = cv.dilate(closing2, kernel, iterations=5)
        sure_bg = cv.morphologyEx(sure_bg, cv.MORPH_OPEN, kernel, iterations=2)

        # sure background area

        contours, h = cv.findContours(sure_fg, 1, cv.CHAIN_APPROX_NONE)                                                  #PREG

        # cv.imshow("sure_fg", sure_fg)
        # cv.imshow("img", img)
        # cv.imshow("sure_bg", sure_bg)

        # Finding unknonwn region
        sure_fg = np.uint8(sure_fg)

        unknown = cv.subtract(sure_bg, sure_fg)
        cv.imshow("Unknown", unknown)

        # Marker labelling
        _, markers = cv.connectedComponents(image=sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv.watershed(img, markers)

        img[markers == -1] = [255, 0, 0]
        cv.imshow("IMAGE", img)

        # cv.drawContours(sure_fg, contours, -1, (0, 0, 255), 2)
        img = cv.putText(img, str(len(contours)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                      2, cv.LINE_AA)
        cv.imshow("numero",img)

        #color
        color_img = color.label2rgb(markers, bg_label=1)
        cv.imshow("seg", color_img)
        map = cv.applyColorMap(grey, cv.COLORMAP_JET)
        cv.imshow("map", map)

        # Area

        _, threshh = cv.threshold(markers.copy().astype(np.uint8), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(threshh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        #cv.waitKey()
        for cnt in contours:
            track_value = 20000
            # track_value = get_area_value()
            image = contour_area(img, cnt, track_value)
        cv.imshow("Area", img.copy())
        #

        if cv.waitKey(0) & 0xFF == ord('x'):
            #
            #markers = markers.astype(np.uint8)
            #_, thresh_marker = cv.threshold(markers, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
            #contours, _ = cv.findContours(thresh_marker, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            #img_contours = img.copy()
            #for cnt in contours:
            #    area = cv.contourArea(cnt)
            #    # if area < valor del thres:
            #    img_contours = cv.drawContours(img_contours, cnt, -1, (0, 0, 255), 4)
            #cv.imshow('img_contours', img_contours)
            #normal(markers)
            #cv.waitKey(0)
            #binary(img,sure_bg,thresh2)
            #binary(img,sure_fg,thresh)

            break






