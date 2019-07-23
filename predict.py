#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:45:23 2019

@author: jaisi8631
"""

# import required dependencies
import config
import numpy as np
import argparse
import imutils
import cv2
import os

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from glob import glob


# create an argument parser for the path to the image to be predicted
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, default = "test-shihtzu.jpg",
                help = "path to test dog image")
args = vars(ap.parse_args())


# load and compile model
model = load_model('model.h5')
opt = SGD(lr = 1e-1, momentum = 0.9, decay = 1e-2 / config.EPOCHS)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])


# read in and preprocess test image from disk
image = cv2.imread(args["image"])
image = cv2.resize(image, (config.imgW, config.imgH))

image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)


# generate predictions on test image
result = model.predict(image)
pred = np.argmax(result, axis=1)
prediction = "UNRECOGNIZABLE"


# cast predictions to dog breed using saved label mapping
label_map = np.load('labels_dictionary.npy').item()
breed = list(label_map)[pred[0]]
data = breed.split("-")
breed = data[1].replace("/", "")
breed = breed.replace("_", " ")


# output prediction
print("The prediction is: " + breed)