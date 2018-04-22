# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 00:57:30 2018

@author: Amir
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D


from keras import backend as K
K.set_image_dim_ordering('th')


def fnBuildModel(preCalculatedWeightPath=None, shape=(48, 48)):
    # In Keras, model is created as Sequential() and more layers are added to build architecture.
    model = Sequential()

    model.add (ZeroPadding2D ((1, 1), input_shape=(1, 48, 48)))
    model.add (Conv2D (128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), input_shape=(1, 48, 48)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add (ZeroPadding2D ((1, 1)))
    model.add (Conv2D (64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    print ("Create model successfully")
    
    if preCalculatedWeightPath:
        model.load_weights(preCalculatedWeightPath)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
        metrics=['accuracy'])
		


    return model
    
    