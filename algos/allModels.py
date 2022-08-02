'''
JMJPFU
20-July-2022
Lord bless this attempt of yours

This is the script to get all the models
'''
from tensorflow.keras.models import load_model
import pickle

class getModels:
    # Start the init methods
    def __init__(self,config):
        self.config = config

    # Below is the method to return a loaded model
    def returnModel(self):
        # Read the model from the path
        Truckmodel = load_model(self.config["MODEL_PATH"])
        # Read the encoder from the path
        Truckencoder = pickle.loads(open(self.config["ENCODER_PATH"], "rb").read())
        return Truckmodel,Truckencoder

