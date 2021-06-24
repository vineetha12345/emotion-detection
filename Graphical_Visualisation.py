import cv2
import numpy as np
from model import FacialExpressionModel

model = FacialExpressionModel("model.json", "model_weights.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def Emotion_Analysis(img):
    path = "static/" + str(img)
    image = cv2.imread(path)
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3,5)
    if len(faces) == 0:
        return [img]
    for (x, y, w, h) in faces:
        fc= gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred=model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, pred, (x, y),font,1, (255,255,0),2)
        cv2.rectangle(image, (x, y), (x+w,y+h), (255, 0, 0), 2)
        path = "static/" + "pred" + str(img)
        cv2.imwrite(path, image)

    return ([img, "pred" + img])
