import cv2
import numpy as np

from hu_gen import hu_moments_of_file, generate_hu_moments_file
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
    if val==1 :
        hu_moments = hu_moments_of_file(filename)  # Genera los momentos de hu de los files de testing
        sample = np.array([hu_moments], dtype=np.float32)  # numpy
        testResponse = model.predict(sample)[1]  # Predice la clase de cada file

        image = cv2.imread(filename)
        image_with_text = cv2.putText(image, int_to_label(testResponse), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                      2, cv2.LINE_AA)
        cv2.imshow("result", image_with_text)
       # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('k'):
        ticks = str(cv2.getTickCount())
        cv2.imwrite(ticks + '.png', frame)
        filename= ticks + '.png'
        val=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# apagamos la capturadora y cerramos las ventanas que se abrieron
cv2.destroyAllWindows()
