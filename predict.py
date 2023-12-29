import cv2
import tensorflow as tf
import numpy as np
import pickle
with open("ResultsMap.pkl", "rb") as f:
    resultMap = pickle.load(f)
filename = 'test/testmul.png'
image = cv2.imread(filename)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
save_model = tf.keras.models.load_model("modeltrained.h5")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fa = face_cascade.detectMultiScale(gray, 1.1, 5)
fontface = cv2.FONT_HERSHEY_SIMPLEX
for (x, y, w, h) in fa:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))
    roi_gray = roi_gray.reshape((100,100,1))
    roi_gray = np.array(roi_gray)
    result = save_model.predict(np.array([roi_gray]))
    print("Đầu ra cuối cùng:")
    print(result)
    print('Nơ ron thứ k có xác xuất lớn nhất: ', np.argmax(result) + 1)
    cv2.imshow('trainning',
               cv2.putText(image, resultMap[np.argmax(result)], (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2))
cv2.waitKey(0)
cv2.destroyAllWindows()
