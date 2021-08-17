import cv2

from Trackbar import get_trackbar_value, create_trackbar
from frame_editor import denoise

alpha_slider_max = 125
window_name = 'Window'
trackbar_name = 'Trackbar'
slider_max = 151
#cv2.namedWindow(window_name)
cap= cv2.VideoCapture(0)
#create_trackbar(trackbar_name, window_name, slider_max)

while True:
    # metodo para que la camara "lea"
    # frame son las "imagenes" (frames) de cada milisegundo
    ret, frame = cap.read()
    # aca las espejamos para que se vean bien
    frame = cv2.flip(frame, 1)
    # aca las mostramos en una ventana
    cv2.imshow('Normal', frame)


    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Grey', grey)

   # trackbar_val = get_trackbar_value(trackbar_name, window_name)

    adapt = cv2.adaptiveThreshold(grey, alpha_slider_max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    #Ese grey antes era binary del metodo q no creamos
    #cv2.namedWindow('trackbar')
   # cv2.createTrackbar('Trackbar', 'trackbar', 0, alpha_slider_max, grey)
    cv2.imshow("Mean", adapt)

    frame_denoised = denoise(adapt, cv2.MORPH_ELLIPSE, 10)
    cv2.imshow("denoised", frame_denoised)







    if cv2.waitKey(1) == ord('h'):
        ticks = str(cv2.getTickCount())
        cv2.imwrite(ticks + '.png', frame)
    # al presionar z salimos del loop
    if cv2.waitKey(1) == ord('z'):
        break

# apagamos la capturadora y cerramos las ventanas que se nos abrieron
cap.release()
cv2.destroyAllWindows()
