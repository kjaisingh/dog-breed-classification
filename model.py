#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:51:50 2019

@author: jaisi8631
"""

# import required dependencies
import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.models import Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D


# define constants
trainDir = config.TRAIN_PATH
valDir = config.VAL_PATH
testDir = config.TEST_PATH


# create data augmentors and generators
TRAIN = len(list(paths.list_images(trainDir)))
VAL = len(list(paths.list_images(valDir)))
TEST = len(list(paths.list_images(testDir)))

trainAug = ImageDataGenerator(rescale = 1./255, fill_mode = "nearest")
valAug = ImageDataGenerator(rescale = 1./255, fill_mode = "nearest")

trainGen = trainAug.flow_from_directory(trainDir,
                                        target_size = (config.imgW, config.imgH),
                                        color_mode = 'rgb',
                                        batch_size = config.BS,
                                        class_mode = 'categorical',
                                        shuffle = True)

valGen = valAug.flow_from_directory(valDir,
                                    target_size = (config.imgW, config.imgH),
                                    color_mode = 'rgb',
                                    batch_size = config.BS,
                                    class_mode = "categorical",
                                    shuffle = False)

testGen = valAug.flow_from_directory(testDir,
                                   target_size = (config.imgW, config.imgH),
                                   color_mode = 'rgb',
                                   batch_size = config.BS,
                                   class_mode = "categorical",
                                   shuffle = False)


# create and save mapping of class labels
label_map = (trainGen.class_indices)
np.save('labels_dictionary.npy', label_map) 


# create model skeleton
base = ResNet50(weights = "imagenet", include_top = False, 
                       input_shape = (config.imgW, config.imgH, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.4)(x)
x = Dense(512, activation = "relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.1)(x)
preds = Dense(config.NB, activation = "softmax")(x)

model = Model(inputs = base.input, outputs = preds)


# indicate trainable and non-trainable layers
for i,layer in enumerate(model.layers):
    print(i,layer.name)

for layer in model.layers[:100]:
    layer.trainable=False
for layer in model.layers[100:]:
    layer.trainable=True
    
model.summary()


# compile and train model
opt = SGD(lr = 1e-1, momentum = 0.9, decay = 1e-2 / config.EPOCHS)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
H = model.fit_generator(
        trainGen,
        epochs = config.EPOCHS,
        steps_per_epoch = TRAIN // config.BS,
        validation_data = valGen,
        validation_steps = VAL // config.BS)


# save model to dis
model.save('model.h5')


# generate and evaluate predictions using model
testGen.reset()
predictions = model.predict_generator(testGen, steps = (TEST // config.BS) + 1) 
predictions = np.argmax(predictions, axis = 1)

print("Test set accuracy: " + 
      str(accuracy_score(testGen.classes, predictions, normalize = True) * 100) 
      + "%") 

print(classification_report(testGen.classes, predictions,
                            target_names = testGen.class_indices.keys())) 