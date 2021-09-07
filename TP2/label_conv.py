def label_to_int(string_label):
    if string_label == '5-point-star': return 3
    if string_label == 'rectangle': return 1
    if string_label == 'circle': return 0
    if string_label == 'lightning': return 2
    if string_label == 'triangle':
        return 4

    else:
        raise Exception('unkown class_label')


def int_to_label(string_label):
    # deberian leer del csv de etiquetas
    if string_label == 0: return 'circle'
    if string_label == 1: return 'rectangle'
    if string_label == 2: return 'lightning'
    if string_label == 3: return '5-point-star'
    if string_label == 4:
        return 'triangle'
    else:
        raise Exception('unkown class_label')
