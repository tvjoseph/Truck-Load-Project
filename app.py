'''
JMJPFU
Lord bless this attempt of yours
27-June-2022

This is the flask app for the inference engine
'''

from tlDetector import Detectormain
from flask import Flask,request
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import flasgger
from flasgger import Swagger



print("[INFO] Starting the Application")
app = Flask(__name__)
Swagger(app)

# Instantiate the Detectormain class
dm = Detectormain()


@app.route('/predict_file',methods = ["POST"])
def infer():
    """ Predicting the type of truck load based on input images
    ---
    parameters:
      - name: truckImage
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The predicted class and probability

    """
    # Get the image object
    rawImg = request.files.get("truckImage")
    # Pass the image object to the inference function
    trgName, prob = dm.inferLocal(image=rawImg)
    # Return the file name and its probability
    return trgName + str(prob)

if __name__ == '__main__':
    app.run()