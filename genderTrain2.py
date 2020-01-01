import os;import sys;import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense,  Dropout, Activation, MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Model,Sequential
pictureDimensions = (218,178);picture3D = (218,178,3);batchSize = 32
rootFolder = "newData\\A2\\";sets = ["train","test","validate"];folderNames = {}
for mySet in sets: folderNames[mySet]= rootFolder + mySet
data_generator = ImageDataGenerator()
train_it = data_generator.flow_from_directory(folderNames["train"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
validate_it = data_generator.flow_from_directory(folderNames["validate"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
test_it = data_generator.flow_from_directory(folderNames["validate"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
STEP_SIZE_TRAIN=train_it.n/batchSize
STEP_SIZE_VALIDATE=validate_it.n/batchSize
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=picture3D));model.add(Activation('relu'));model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)));model.add(Activation('relu'));model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)));model.add(Activation('relu'));model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten());model.add(Dense(64));model.add(Activation('relu'));model.add(Dropout(0.5))
model.add(Dense(1));model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit_generator(generator=train_it,steps_per_epoch=STEP_SIZE_TRAIN,epochs=1,
					validation_data=validate_it,validation_freq = 1, validation_steps=STEP_SIZE_VALIDATE)
model.save("Gender.h5")