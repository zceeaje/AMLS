import os;import sys;import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense,  Dropout, Activation
from keras.models import Model,Sequential
from keras.layers import MaxPooling2D, AveragePooling2D, Conv2D

																																						
pictureDimensions = (218,178);picture3D = (218,178,3);batchSize = 32
imageFolder = r"Data\A1"

data_generator = ImageDataGenerator()
train_it = data_generator.flow_from_directory(imageFolder,target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
STEP_SIZE_TRAIN=train_it.n/batchSize

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=picture3D))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=train_it,steps_per_epoch=STEP_SIZE_TRAIN,epochs=10)

