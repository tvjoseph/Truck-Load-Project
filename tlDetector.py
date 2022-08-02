'''
JMJPFU
Lord bless this attempt of yours
27-June-2022

This is the main file for the truck load detection application
'''


from utils import Conf
import numpy as np
from processes import Inference,tlVideoProcess
import cv2
from PIL import Image
from pathlib import Path
import tempfile
from algos import getModels
from utils.helperFuncs import most_frequent


class Detectormain:
    def __init__(self):
        # Define the path of the configuration file
        confPath = "config/tlConfig.json"
        # Load the configuration file
        self.conf = Conf(confPath)
        print('config',self.conf)
        # Initialise the Inference class
        self.inf = Inference(self.conf)
        # Initiate the video process class
        self.vp = tlVideoProcess(self.conf)
        # Get the algorithm class
        self.algo = getModels(self.conf)
        # GEt the model and binarizer
        self.model,self.binarizer = self.algo.returnModel()

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

    def inferVideo(self,videoFile,videoCapture=False):
        '''
        videoCapture : This is a flag which indicates whether we would be processing using video file or live video stream
        '''

        if videoCapture:
            # If the processing is through camera the videofile will be None
            self.vp.frameCapture(videoCapture,None)
        else:

            with tempfile.TemporaryDirectory() as td:
                temp_filename = Path(td) / 'uploaded_video'
                videoFile.save(temp_filename)

                print("Temporary video file path",temp_filename)

                # Return all the trackers and the string for further processing.
                retVal,allTrackers = self.vp.frameCapture(videoCapture, str(temp_filename))
                # Next we start the process of truck type detection from the video dictionary
                detectedObjs = self.objTypeDetect(allTrackers)


                return str(detectedObjs)

    def objTypeDetect(self,allTrackers):
        """
        allTrackers : This is the dictionary containing all the object which have been tracked
        return : This function returns another dictionary containing the object which is detected and the class of the object
        """
        # Initialise dictionary to store all the objects and classes
        detectedObjs  = {}
        # Get the track dictionary for further analysis
        trackDic = self.trackDicMaker(allTrackers)

        # Get the type of object from each trackDic
        for key in trackDic.keys():
            print("[INFO] Starting the prediction process")
            # For each key detect the objects using the function
            objClass = self.objClassifier(trackDic[key],key)
            # Store the detected object against the object ID
            detectedObjs[key] = objClass
        return detectedObjs

    ############# The below function needs to go to Utility module #################

    def trackDicMaker(self,allTrackers):
        """
        allTrackers : This is the list of dictionaries containing the tracked objects
        """
        trackDic = {}
        for i, trck in enumerate(allTrackers):
            if len(trck) == 0:
                continue
            for dic in trck:
                if dic["trackID"] in trackDic.keys():
                    trackDic[dic["trackID"]]['box'].append(dic["box"])
                    trackDic[dic["trackID"]]['img'].append(dic["img"])
                else:
                    trackDic[dic["trackID"]] = {'box': [], 'img': []}
                    trackDic[dic["trackID"]]['box'].append(dic["box"])
                    trackDic[dic["trackID"]]['img'].append(dic["img"])
        # Return the tracked dictionary
        return trackDic
    ###############################################################################################################
    def objClassifier(self,objDic,key):
        """
        objDic : This is the object dictionary which is passed from the earlier function.
        key : This is the object tracking ID
        Returns : The predicted class of the cropped object
        """
        objClass = []
        objVideo = []
        # Load the
        for i, img in enumerate(objDic['img']):
            # Get the coordinates of the box
            left, top, right, bottom = objDic['box'][i]
            # Crop the image based on the coordinates
            crop = img[top:bottom, left:right]
            #print("Crop shape",crop.shape)
            # Store all the cropped images for making video
            objVideo.append(crop)
            # Reshape the image
            tst_roi = cv2.resize(crop, (self.conf["imgResize"], self.conf["imgResize"]), interpolation=cv2.INTER_CUBIC)
            # Expand dimensions of the image
            ex_img = np.expand_dims(tst_roi, axis=0)
            # Do the object classifier for the cropped image and append the class in the dictionary
            preds = self.model.predict(ex_img)
            # For each prediction we need to find the index with maximum probability and store in the class list
            objClass.append(int(list(np.argmax(preds, axis=1))[0]))
            # Create a video file with all the images which are collected.

        # Get the most occuring class
        predClass = self.conf["targets"][most_frequent(objClass)]
        print("[INFO] Starting the video creation process")
        self.videoCreator(objVideo,key,predClass)

        return predClass

    def videoCreator(self,objVideo,key,predClass):
        """
        objVideo ; This is the list of all the cropped images for creating the video
        key : This is the tracker Id for saving the video
        return : Saves the video file in a folder
        int(self.conf["imgResize"]),int(self.conf["imgResize"])
        """
        out = cv2.VideoWriter(self.conf["outputVideo"] + str(key) +'_' + predClass +'.mp4',cv2.VideoWriter_fourcc(*"mp4v"), 15, (self.conf["videoSize"],self.conf["videoSize"]))
        for img in objVideo:
            # Resize the cropped image so as to write to video
            resized = cv2.resize(img,(self.conf["videoSize"], self.conf["videoSize"]), interpolation=cv2.INTER_AREA)
            out.write(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        out.release()
        print("[INFO] completed creating the video for object ID :{}".format(key))










