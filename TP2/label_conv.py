import csv


def label_to_int(string_label):
    with open('shapes/etiquetas.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[1] == string_label: return row[0]


def int_to_label(int_label):
    with open('shapes/etiquetas.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if int(float(row[0])) == int_label: return row[1]
