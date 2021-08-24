import cv2

from contour import get_contours, get_biggest_contour, compare_contours, draw_contours
from trackbar import get_trackbar_value, create_trackbar, adaptive_threshold
from frame_editor import denoise
import random

alpha_slider_max = 125
window_name = 'Denoised'
trackbar_name = 'Trackbar'
slider_max = 17
window_name_2 = 'Mean'
trackbar_name_2 = 'Trackbar2'
slider_max_2 = 151

cv2.namedWindow(window_name)
cv2.namedWindow(window_name_2)

cap = cv2.VideoCapture(0)

create_trackbar(trackbar_name, window_name, slider_max)
create_trackbar(trackbar_name_2, window_name_2, slider_max_2)
biggest_contour = None

color_white = (255, 255, 255)
saved_contours = []
name = random.randint(0,10)

while True:
    # metodo para que la camara "lea", frame son las "imagenes" de cada milisegundo
    ret, frame = cap.read()
    # aca las espejamos para que se vean bien
    frame = cv2.flip(frame, 1)
    # cv2.imshow('Normal', frame)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Grey', grey)

    trackbar_val2 = get_trackbar_value(trackbar_name_2, window_name_2)
    adapt = adaptive_threshold(grey, alpha_slider_max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, trackbar_val2)
    cv2.imshow(window_name_2, adapt)

    trackbar_val = get_trackbar_value(trackbar_name, window_name)

    frame_denoised = denoise(adapt, cv2.MORPH_ELLIPSE, trackbar_val)
    # cv2.imshow(window_name, frame_denoised)
    contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        biggest_contour = get_biggest_contour(contours=contours)
        if compare_contours(contour_to_compare=biggest_contour, saved_contours=saved_contours, max_diff=0.5):
            (x, y), radius = cv2.minEnclosingCircle(biggest_contour)
            center = (int(x), int(y))
            strName = str(name)
            frame_denoised = cv2.putText(frame_denoised, strName, center, cv2.FONT_HERSHEY_SIMPLEX,
                                1, color_white, 2, cv2.LINE_AA)
            draw_contours(frame=frame_denoised, contours=biggest_contour, color=color_white, thickness=20)
        draw_contours(frame=frame_denoised, contours=biggest_contour, color=color_white, thickness=3)

    # cv2.imshow('Window', frame_denoised)
    cv2.imshow(window_name, frame_denoised)

    if cv2.waitKey(1) & 0xFF == ord('k'):
        if biggest_contour is not None:
            saved_contours.append(biggest_contour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# apagamos la capturadora y cerramos las ventanas que se abrieron
cv2.destroyAllWindows()
