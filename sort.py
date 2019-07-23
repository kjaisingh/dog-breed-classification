#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:17:07 2019

@author: jaisi8631
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:14:12 2018

@author: jaisi8631
"""

# import required dependencies
from imutils import paths
import config
import random
import shutil
import os


# create list of image files and shuffle
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)


# split dataset based on specified train-test split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]


# use part of training set as validation set
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]


# create list, called datasets, of three tuples with information
datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH)
]


# loop over each dataset
for(dType, imagePaths, baseOutput) in datasets:
    print("Building '{}' split".format(dType))
    
    # if output directory doesn't exist, create it
    if not os.path.exists(baseOutput):
        print("Creating the '", baseOutput, "' directory")
        os.makedirs(baseOutput)
        
    # loop over images to be placed into dataset
    for inputPath in imagePaths:
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]
        
        # create output path
        labelPath = os.path.sep.join([baseOutput, label])
        
        # if path does not exist, create it
        if not os.path.exists(labelPath):
            print("Creating the '", labelPath, "' directory")
            os.makedirs(labelPath)
        
        # construct the path, and copy the image into the path
        p = os.path.sep.join([labelPath, filename])
        shutil.move(inputPath, p)