import math

import cv2
import numpy
import numpy as np

from hu_gen import hu_moments_of_file, generate_hu_moments_file, get_hu_moments_of_frame
from label_conv import int_to_label
from training_model import train_model


val=0
filename= None
cap = cv2.VideoCapture(0)

generate_hu_moments_file()
model = train_model()

while True:
    # metodo para que la camara "lea", frame son las "imagenes" de cada milisegundo
    ret, frame = cap.read()
    # aca las espejamos para que se vean bien
    frame = cv2.flip(frame, 1)
    cv2.imshow('Normal', frame)
    #if val==1 :
   # hu_moments = get_hu_moments_of_frame(frame)  # Genera los momentos de hu de los files de testing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin  # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
    shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima
    frame_contour = cv2.drawContours(frame, [shape_contour], -1, (255, 0, 0), 3)
    cv2.imshow('contour', frame)
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)  # momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(
            abs(huMoments[i]))  # Mapeo para agrandar la escala.
    hu_moments= huMoments

    sample = np.array([hu_moments], dtype=np.float32)  # numpy
    testResponse = model.predict(sample)[1]  # Predice la clase de cada file
    print(testResponse)
    image_with_text = cv2.putText(frame, int_to_label(testResponse), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                   2, cv2.LINE_AA)
    cv2.imshow("result", image_with_text)


    #if cv2.waitKey(1) & 0xFF == ord('k'):
        #ticks = str(cv2.getTickCount())
        #cv2.imwrite(ticks + '.png', frame)
        #filename= ticks + '.png'
        #if val==1 :
        #    print(hu_moments)
        #    print(testResponse)
        #    print()
        #val=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# apagamos la capturadora y cerramos las ventanas que se abrieron
cv2.destroyAllWindows()
