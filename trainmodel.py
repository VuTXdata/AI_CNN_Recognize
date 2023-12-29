TrainingImagePath = 'data/facegray'
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1/255,
        validation_split=0.2
)
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        color_mode='grayscale',
        target_size=(100, 100),
        class_mode='categorical',
        subset='training'
)
val_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        color_mode='grayscale',
        target_size=(100, 100),
        class_mode='categorical',
        subset='validation'
)
TrainClasses = training_set.class_indices
ResultMap = {}
for faceValue, faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue] = faceName
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
print("Mapping of Face and its ID\n", ResultMap)
OutputNeurons = len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


Model = Sequential()
shape = (100,100, 1)
Model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=shape))
Model.add(Conv2D(32,(3,3), padding="same",activation="relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Conv2D(64,(3,3), padding="same",activation="relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Flatten())
Model.add(Dense(512,activation="relu"))
Model.add(Dense(10,activation="softmax"))
Model.summary()
Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("start training")
Model.fit(training_set,validation_data=val_set,batch_size=8, epochs=100)
Model.save("modeltrained.h5")
