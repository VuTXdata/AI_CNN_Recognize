from PIL import Image
import numpy as np
import cv2
import pickle
import  tensorflow as tf
with open("ResultsMap.pkl", "rb") as f:
    resultMap = pickle.load(f)
save_model = tf.keras.models.load_model("modeltrained.h5")
# for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# size of the image to predict
image_width = 100
image_height = 100

# load the trained model


# the labels for the trained model


# default webcam
stream = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    (grabbed, frame) = stream.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    for (x, y, w, h) in faces:
        roi_rgb = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(src=roi_gray, dsize=(100, 100))
        roi_gray = roi_gray.reshape((100, 100, 1))
        roi_gray = np.array(roi_gray)
        result = save_model.predict(np.array([roi_gray]))


        # Display the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = resultMap[np.argmax(result)]
        color = (255, 0, 255)
        stroke = 2
        cv2.putText(frame, f'({name})', (x,y-8),
            font, 1, color, stroke, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# Cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)