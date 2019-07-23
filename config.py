#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:19:58 2019

@author: jaisi8631
"""

# import required dependencies
import os

# base paths
BASE_PATH = "data"
ORIG_INPUT_DATASET = "stanford-dogs-dataset/Images"

# training, validation and testing set paths
TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
VAL_PATH = os.path.sep.join([BASE_PATH, "val"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

# training split
TRAIN_SPLIT = 0.9

# the amount of validation data as percentage of the training data
VAL_SPLIT = 0.1

# defining constants for the network training process
imgW, imgH = 224, 224
NB = 120
BS = 64
EPOCHS = 20