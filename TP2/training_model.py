import cv2
import csv

import numpy as np

from label_conv import label_to_int

trainData = []
trainLabels = []

# Agarro las cosas en los archivos las guardo en variables y las mando a train data y labels
def load_training_set():
    global trainData
    global trainLabels
    with open('shapes/descriptores.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # sacar ep primer elemento (numero de foto)

            numero_de_foto = row.pop(0)  # saca el primer elemento de la lista
            photo_numb =  int(float(numero_de_foto))# castearlo a int

            # pasar de numero de foto a numero de label

            numero_de_label=convert_to_label_number(photo_numb)


            # appender el numero de label a train label
            # agarrar hu moments y appendearlos a train data


            floats = []
            for n in row:
                floats.append(float(n)) # tiene los momentos de Hu transformados a float.
            trainData.append(np.array(floats, dtype=np.float32)) # momentos de Hu
            trainLabels.append(np.array([numero_de_label], dtype=np.int32)) # Resultados
            #Valores y resultados se necesitan por separados
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)

# transforma los arrays a arrays de forma numpy


def convert_to_label_number(photo_numb):
    with open('shapes/supervision.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if (int(float(row[0])) == photo_numb):
                return int(float(row[1]))


# llama la funcion de arriba, se manda a entrenar y devuelve el modelo entrenado
def train_model():
    load_training_set()

    tree = cv2.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(2)
    tree.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)
    return tree
