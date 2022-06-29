'''
JMJPFU
Lord bless this attempt of yours
27-June-2022

This is the main file for the truck load detection application
'''

import pickle
from utils import Conf
import glob
import pandas as pd
import numpy as np
import os
from processes import Inference
import cv2
from flask import Flask,request
from PIL import Image

class Detectormain:
    def __init__(self):
        # Define the path of the configuration file
        confPath = "config/tlConfig.json"
        # Load the configuration file
        self.conf = Conf(confPath)
        print('config',self.conf)
        # Initialise the Inference class
        self.inf = Inference(self.conf)

    ######################### Inference cycle ##############################
    def inferLocal(self,image=None):
        '''
        This is the method to to inference from files loaded from the configuration files and also from flask
        :param image: Image object as byte files to be converted to opencv format
        :return: Probability and the class name
        '''
        print("[INFO Starting the Inference cycle ]")
        if self.conf["imgLoad"]:
            # Get the prediction score and class by loading the image from configuration file
            print("Loading image from configuration file ")
            prob,trg = self.inf.infGenerator()
        else:
            # Get the prediction by providing an image
            print("Providing an image as the input")
            # Open image using PIL file
            img = Image.open(image.stream)
            # Convert the image object to array
            nimg = np.array(img)
            # Convert the array to opencv format
            ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            # Get the predictions for the input image
            prob, trg = self.inf.infGenerator(image = ocvim)
        # Get the target Name
        trgName = self.conf["targets"][int(trg)]
        print("The image shows a {} with probability {}%".format(trgName,prob[0]*100))
        return trgName,prob





