import cv2
from utils import get_connected_components


def binary(img,grey, thresh):
    map = cv2.applyColorMap(grey, cv2.COLORMAP_JET)

    get_connected_components(thresh, 8, img)
    cv2.imshow("map", map)
    cv2.waitKey()


def normal(img):
    cv2.imshow("normal", img)
    map = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imshow("celulas", map)
    cv2.waitKey()