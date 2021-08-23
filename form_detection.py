import cv2

from Contour import get_contours, get_biggest_contour, compare_contours, draw_contours
from Trackbar import get_trackbar_value, create_trackbar, adaptive_threshold, on_trackbar
from frame_editor import denoise

alpha_slider_max = 125
window_name = 'denoised'
trackbar_name = 'Trackbar'
slider_max = 151

cv2.namedWindow(window_name)
cv2.namedWindow('Mean2')

cap= cv2.VideoCapture(0)

create_trackbar(trackbar_name, window_name, 17)
create_trackbar('Trackbar2', 'Mean2', slider_max)
biggest_contour = None

color_red = (255, 0, 0)
saved_contours = []

while True:
    # metodo para que la camara "lea"
    # frame son las "imagenes" (frames) de cada milisegundo
    ret, frame = cap.read()
    # aca las espejamos para que se vean bien
    frame = cv2.flip(frame, 1)
    # aca las mostramos en una ventana
    cv2.imshow('Normal', frame)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grey', grey)

    trackbar_val2 = get_trackbar_value('Trackbar2', 'Mean2')
    adapt2 = adaptive_threshold(grey, alpha_slider_max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, trackbar_val2)
    cv2.imshow("Mean2", adapt2)

    trackbar_val = get_trackbar_value(trackbar_name, window_name)
   # adapt = adaptive_threshold(grey, alpha_slider_max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, trackbar_val)
    #cv2.imshow("Mean", adapt)

    frame_denoised = denoise(adapt2, cv2.MORPH_ELLIPSE, trackbar_val)
    cv2.imshow("denoised", frame_denoised)
    contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        biggest_contour = get_biggest_contour(contours=contours)
        # hu_moments = get_hu_moments(contour=biggest_contour)
        if compare_contours(contour_to_compare=biggest_contour, saved_contours=saved_contours, max_diff=0.5):
            draw_contours(frame=frame_denoised, contours=biggest_contour, color=color_red, thickness=20)
        draw_contours(frame=frame_denoised, contours=biggest_contour, color=color_red, thickness=3)

    cv2.imshow('Window', frame_denoised)
    if cv2.waitKey(1) & 0xFF == ord('k'):
        if biggest_contour is not None:
            # save_moment(hu_moments=hu_moments, file_name="hu_moments.txt")
            saved_contours.append(biggest_contour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# apagamos la capturadora y cerramos las ventanas que se nos abrieron
cv2.destroyAllWindows()
