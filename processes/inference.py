'''
JMJPFU
Lord bless this attempt of yours
27-June-2022

This is the file for the inference cycle
'''

from numpy import array
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import os
from data import DataProcessor
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import pickle

class Inference:
    # Adding the initiator
    def __init__(self,configfile):
        # This is the first method in the trainProcess class
        self.config = configfile
        self.dp = DataProcessor(self.config)
        # Load the model
        self.model = load_model(self.config["MODEL_PATH"])
        # load the class encoder
        self.lb = pickle.loads(open(self.config["ENCODER_PATH"], "rb").read())
        # Get the target names
        # Converting the target names as string for classification report
        self.target_names = list(map(str, self.lb.classes_))

    # Below is the method to process the input data for inference
    def infDataProcessor(self,image=False,imgPath=True):
        # The image is converted to an array
        roi = self.dp.roiExtractor(image,imgPath)
        # Expand the dimensions of the image
        testImage = np.expand_dims(roi, axis=0)
        return testImage
    # Below is the method for generating predictions from the image

    def infGenerator(self,image=False):
        # Get the test image processed
        try :
            print("image shape",image.shape)
            img = self.infDataProcessor(image,False)
        except:
            print("Getting into the exception loop")
            img = self.infDataProcessor()
        # Generate the prediction
        predictions = self.model.predict(img)
        # Get the argmax of the predictions
        # For each prediction we need to find the index with maximum probability
        predIdxs = np.argmax(predictions, axis=1)
        # Get the target name also
        trgname = self.target_names[predIdxs[0]]
        # Return the prediction probability and target name
        return predictions[0][predIdxs],trgname




