import cv2 as cv


def get_trackbar_area():
    cv.namedWindow('Area')
    cv.createTrackbar('Window', 'Area', 2500, 5000, on_Change)


def on_Change():
    pass


def get_area_value():
    return int(cv.getTrackbarPos('Window', 'Area')/2)*2 + 3


def contour_area(img, cnt, thresh_value=2500):
    area = cv.contourArea(cnt)
    if area < thresh_value:
        x, y, w, h = cv.boundingRect(cnt)
        cv.putText(img, str(int(area)), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return cv.drawContours(img, cnt, -1, (0, 255, 0), 5)
    else:
        # cv.addText()
        return cv.drawContours(img, cnt, -1, (0, 0, 255), 10)

    # cv.putText(img, area, (x + int(w / 10), y + int(h / 2)), cv.FONT_HERSHEY_SIMPLEX, 0.7,
    #            (255, 255, 255), 2)
