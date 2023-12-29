import cv2
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot
from tensorflow._api.v2.compat.v2 import nn

from tensorflow.keras.models import Model

filename = 'data/facegray/hoanganhtu/hoanganhtu_3.png'
image = cv2.imread(filename)
model = tf.keras.models.load_model("modeltrained.h5")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
roi_gray = cv2.resize(src=gray, dsize=(100, 100))
roi_gray = roi_gray.reshape((100, 100, 1))
roi_gray = np.array(roi_gray)

for i in range(len(model.layers)):
    layer = model.layers[i]
    print(i , layer.name , layer.output.shape)
model = Model(inputs=model.inputs , outputs=model.layers[4].output)
features = model.predict(np.array([roi_gray]))
print(features)
fig = pyplot.figure(figsize=(20, 15))
for i in range(1, features.shape[3] + 1):
    pyplot.subplot(8, 8, i)
    pyplot.imshow(features[0, :, :, i - 1], cmap='gray')

pyplot.show()