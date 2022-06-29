# +

'''
This is the script for preprocessing data
'''
import sys
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import random
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Defining the Data Processer class, where all preprocessing steps are added
class DataProcessor:
    def __init__(self,configfile):
        # This is the first method in the DataProcessor class
        # Initialise configfile,ws and separate multivariate processors
        self.config = configfile

        
    # This is the method to process the raw images
    # Functions to extract the bounding boxes and preprocess the image
    def roiExtractor(self,img = False,imgPath=True):
        if imgPath:
            # Get the path from the config file
            path = self.config["infPath"]
            # Read the image
            img = cv2.imread(path)
        # Reshape the image
        img = cv2.resize(img, (int(self.config["imgResize"]), int(self.config["imgResize"])), interpolation=cv2.INTER_CUBIC)
        # Convert the image to an array
        img = img_to_array(img)
        # Preprocess the image
        img = preprocess_input(img)
        return img
