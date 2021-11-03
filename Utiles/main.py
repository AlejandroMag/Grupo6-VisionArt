import cv2 as cv
import numpy as np
from skimage import color
from area import get_area_value, contour_area, get_trackbar_area

image = cv.imread('C:/Users/owner/projects/faculty/vision-artificial/TPs/proyecto3/levadura.png')
image = cv.resize(image, None, fx = 0.60, fy = 0.60, interpolation=cv.INTER_CUBIC)
cv.imshow('Levadura', image)
bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# get_trackbar_area()


def binary(val):
    ret, thresh = cv.threshold(bw, val, 255, cv.THRESH_BINARY_INV)
    cv.imshow("Thresh", thresh)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, (3, 3))
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, (3, 3))
    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv.contourArea(cnt) < 350:
            contours.remove(cnt)
    x, y, w, h = 0, 0, 175, 75
    cv.rectangle(bw, (x, x), (x + w, y + h), (0, 0, 0), -1)
    cv.putText(bw, str(len(contours)), (x + int(w / 10), y + int(h / 2)), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)


def connected_components():
    global image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)
    cv.imshow("binary", thresh)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1)
    cv.imshow("closing", closing)
    # lo que seguro es fondo
    sure_bg = cv.dilate(closing, kernel, iterations=5)
    sure_bg = cv.morphologyEx(sure_bg, cv.MORPH_OPEN, kernel, iterations=2)
    cv.imshow("sure background", sure_bg)
    cv.waitKey()
    # lo que seguro es frente
    _, sure_fg = cv.threshold(gray, 85, 255, cv.THRESH_BINARY)
    sure_fg = cv.dilate(sure_fg, kernel, iterations=4)
    cv.imshow("sure foreground", sure_fg)
    cv.waitKey()
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    # Hacer que el fondo sea 1
    markers = markers + 1
    # marcamos lo desconocido como 0
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    cv.imshow("img", image)
    color_img = color.label2rgb(markers, bg_label=1)
    cv.imshow("seg", color_img)
    map = cv.applyColorMap(gray, cv.COLORMAP_JET)
    cv.imshow("map", map)

    # Clasificar por area
    _, thresh = cv.threshold(markers.copy().astype(np.uint8), 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.waitKey()
    for cnt in contours:
        track_value = 3000
        # track_value = get_area_value()
        image = contour_area(image, cnt, track_value)
    cv.imshow("Area", image.copy())


cv.namedWindow('Thresh')
cv.createTrackbar('Trackbar', 'Thresh', 0, 255, binary)

binary(0)
connected_components()

cv.waitKey()
