import cv2
import csv
import glob
import numpy
import math

aux = 0

def generate_hu_moments_file():
    with open('shapes/descriptores.csv', 'w', newline='') as file:  # Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        # Ahora escribo los momentos de Hu de cada uno de las figuras. Con el string "rectangle...etc" busca en la carpeta donde estan cada una de las imagenes
        # generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
        write_hu_moments("imagenes", writer,aux)


# Escribo los valores de los momentos de Hu en el archivo

def write_hu_moments(folder, writer, aux):
    files = glob.glob('./shapes/' + folder + '/*')  # label recibe el nombre de la carpeta
    hu_moments = []
    for index, file in enumerate(files):
        hu_moments.append(hu_moments_of_file(file))
        for mom in hu_moments:
            flattened = mom.ravel()
        # paso de un array de arrays a un array simple.
#            row = numpy.insert(flattened, 0, int(aux))
        row = numpy.insert(flattened, 0, str(int(index+1)))

        # row = numpy.append(flattened, picNumber)  # le metes el flattened array y le agregas el label
        writer.writerow(row)  # Escribe una linea en el archivo.

# Encargada de generar los momentos de Hu para las imagenes

def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
    shape_contour = max(contours, key=cv2.contourArea) # Agarra el contorno de area maxima
        # Calculate Moments
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)  # momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i])) # Mapeo para agrandar la escala.
    return huMoments



def get_hu_moments_of_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8)  # Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos
    shape_contour = max(contours, key=cv2.contourArea) # Agarra el contorno de area maxima
    frame_contour = cv2.drawContours(frame, [shape_contour], -1, (255,0,0), 3)
    cv2.imshow('contour', frame_contour)
    cv2.waitKey(0)
    # Calculate Moments
    moments = cv2.moments(shape_contour)  # momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)  # momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i])) # Mapeo para agrandar la escala.
    return huMoments
