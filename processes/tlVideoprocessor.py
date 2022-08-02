'''
JMJPFU
1-July-2022
Lord bless this attempt of yours

This is the process for analysing video of truck loads
'''


import cv2
import time
import yolov5
from operator import itemgetter
import pickle
import os
from scipy.optimize import linear_sum_assignment
import numpy as np
from math import sqrt, exp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy
import glob



class ObjectTracker():
    def __init__(self, idx, box,frame,age=1, unmatched_age=0):
        """
        Init function. The id and box of the object which is tracked
        """
        self.idx = idx
        self.box = box
        self.frame = frame
        self.age = age
        self.unmatched_age = unmatched_age


class tlVideoProcess:
    def __init__(self,conf):
        '''
        conf : This is the config file
        '''
        self.conf = conf
        self.yolo = yolov5.load(self.conf["yoloPath"])
        self.yolo.conf = self.conf["yoloConf"]
        self.yolo.iou = self.conf["yoloIou"]
        classesFile = self.conf["cocoNames"]
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        self.classes = classes
        # Define the tracking variables
        self.out_imgs = []
        self.idx = 0
        self.stored_obstacles = []
        self.MAX_UNMATCHED_AGE = self.conf["MAX_UNMATCHED_AGE"]
        self.MIN_HIT_STREAK = self.conf["MIN_HIT_STREAK"]

    # Below is the method to capture video frames either from video file or video camera
    def frameCapture(self,videoCapture,videoFile):
        '''
        videoCapture : Flag indicating whether to process live video or no
        videoFile : THis is the video object which needs to be passed for processing
        '''
        # Initialize the video stream and allow the video stream to warm up
        print('[INFO] warming up the camera ....')
        # Checking if it is a video or camera
        if videoCapture:
            camera = cv2.VideoCapture(0)
            time.sleep(2.0)
        else:
            # Capture the frames from the video
            camera = cv2.VideoCapture(videoFile)
            # Initialise a list to catpure all category dictionaries
            alldicList = []
            # Initialise a list to capture only the objects we want
            objdicList = []
            # Start a list to store the tracking class
            allTrackers = []
            # Start a list to capture the final images
            allImgs = []
            # Start a counter to store different values with counter name
            counter = 0
            while True:
                success, image = camera.read()
                # If we are viewing a video and did not grab a frame
                if not success:
                    print('Video completed')
                    break
                # Start the video analysis processes which includes frame detection, frame classification if truck etc
                ###### Detection using Yolo ##############
                # Get the indexes and categories filtered for the objects to be detected
                categoryDic = self.frameDetector(image)
                # Append the dictionary to the list if an object is found
                if len(categoryDic) > 0:
                    #print("[INFO] Detected an object")
                    # Take the bounding box info of only the objects we want
                    objdicList,objDic = self.objExtractor(categoryDic, image, counter, objdicList)
                    ## Start the object tracking between successive frames
                    #print("printing the object diclist before tracking",objDic)
                    out_img, out_stored_obstacles = self.Trackmain(image, objDic)
                    #print("Printing out stored obstacles",out_stored_obstacles)
                    # Save the tracked class information in the list
                    allTrackers.append(out_stored_obstacles)
                    # Save the out images
                    allImgs.append(out_img)
                    # Save the complete list of objects detected
                    alldicList.append(categoryDic)
                    #print("length of all dicList",len(alldicList))
                if len(alldicList) > 500:
                    print("Breaking the list with 750 elements")
                    break
                # Increment the counter
                counter += 1
            # Save the list in a pickle file
            with open('objects.pkl', 'wb') as f:
                pickle.dump(alldicList, f)
            with open('Selectobjects.pkl', 'wb') as f:
                pickle.dump(objdicList, f)
            with open('objectTracker.pkl', 'wb') as f:
                pickle.dump(allTrackers, f)
            with open('outImages.pkl','wb') as f:
                pickle.dump(allImgs,f)
            return str(objdicList),allTrackers

    # This is the method to take frames and then return only the object we want
    def frameDetector(self,image):
        '''
        image : This is the frame which needs to be analysed
        '''
        # Get the results after predictions
        results = self.yolo(image)
        # Get the predictions from the results
        predictions = results.pred[0]
        # Get the boxes, scores and categories
        boxes = predictions[:, :4].tolist()
        scores = predictions[:, 4].tolist()
        categories = predictions[:, 5].tolist()
        # Convert the boxes into integers
        boxes_int = [[int(v) for v in box] for box in boxes]
        categories_int = [int(c) for c in categories]
        # Start analysis only if there are any objects to detect
        if len(boxes_int) > 0:
            # Get the objects in the categories and in a dictionary
            categoryDic = self.categoryCapture(boxes_int,categories_int)
            return categoryDic
        else:
            return {}
    # The below method is to capture the objects and its bounding boxes in a dictionary
    def categoryCapture(self,boxes,categories):
        '''
        boxes : The bounding boxes for the objects
        categories : The class names for the objects
        '''
        # initialise a dictionary to capture the objects
        categoryDic = {}
        # First find the unique classes in the categories
        allClasses = list(set(categories))
        # Loop through each class and store those values in a dictionary
        for obj in allClasses:
            # Find the indexes where the categories are what we want
            catIndexes = [i for i, x in enumerate(categories) if x == obj]
            # Get the boxes for the filtered indexes
            filtBoxes = itemgetter(*catIndexes)(boxes)
            # Store the information in the dictionary
            categoryDic[self.classes[obj]] = filtBoxes
        return categoryDic
    # This is the method to take the bounding boxes of the object of interest and crop the image
    def objExtractor(self,categoryDic,image,counter,objdicList,crop=False):
        """
        categoryDic ; This is the dictionary with the detected objects and its bounding boxes
        image : The image which needs to be cropped with the bounding boxes
        counter : This is a sample counter just to store image with an id
        objdicList : This is a list to store all the object dictionaries
        crop ; This is a flag to initiate cropping of the images based on the bounding boxes
        return : The images of the objects we want
        """
        # Initialise another dictionary to store only the required bounding boxes
        objDic = {}
        matches = [i for i in self.conf["objects"] if i in list(categoryDic.keys())]
        if len(matches) > 0:
            for obj in matches:
                print("[INFO] detected a {}".format(obj))
                # Get all the bounding boxes for the object
                boxes = categoryDic[obj]
                # Check if there are multiple boxes or not
                if type(boxes) == tuple:
                    # Convert the tuple  to a list of lists
                    boxes = list(boxes)
                else:
                    # Convert the single list into a list of lists
                    boxes = [boxes]
                # Store the matching object into a new dictionary
                objDic[obj] = boxes
                if crop:
                    # Start the cropping process only if the flag is true
                    # Go through each box to extract multiple objects if any
                    for i,box in enumerate(boxes):
                        # Take the roi of the image based on the bounding box
                        objRoi = image[box[1]:box[3], box[0]:box[2]]
                        # Save the image to verify the detection process
                        p = str(counter)+str(i)+"{}.png".format(obj)
                        # Write the file in the folder
                        cv2.imwrite(p, objRoi)
            # Store this dictionary into the object list
            objdicList.append(objDic)
        return objdicList,objDic

    # THis is the method to calculate the IOU between boxes
    def box_iou(self,box1, box2, w=1280, h=360):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)  # abs((xi2 - xi1)*(yi2 - yi1))
        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)  # abs((box1[3] - box1[1])*(box1[2]- box1[0]))
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)  # abs((box2[3] - box2[1])*(box2[2]- box2[0]))
        union_area = (box1_area + box2_area) - inter_area
        # compute the IoU
        iou = inter_area / float(union_area)
        return iou

    # This is a utility function to check division
    def check_division_by_0(self,value, epsilon=0.01):
        if value < epsilon:
            value = epsilon
        return value

    # This is the method to calculated IOU using Sanchez Matilla method
    def sanchez_matilla(self,box1, box2, w=1280, h=360):
        Q_dist = sqrt(pow(w, 2) + pow(h, 2))  # First real-life Pythagore use in your life
        Q_shape = w * h
        distance_term = Q_dist / self.check_division_by_0(sqrt(pow(box1[0] - box2[0], 2) + pow(box1[1] - box2[1], 2)))
        shape_term = Q_shape / self.check_division_by_0(sqrt(pow(box1[2] - box2[2], 2) + pow(box1[3] - box2[3], 2)))
        linear_cost = distance_term * shape_term
        return linear_cost

    # THis is the method to calculate IOU using Yu method
    def yu(self,box1, box2):
        w1 = 0.5
        w2 = 1.5
        a = (box1[0] - box2[0]) / self.check_division_by_0(box1[2])
        a_2 = pow(a, 2)
        b = (box1[1] - box2[1]) / self.check_division_by_0(box1[3])
        b_2 = pow(b, 2)
        ab = (a_2 + b_2) * w1 * (-1)
        c = abs(box1[3] - box2[3]) / (box1[3] + box2[3])
        d = abs(box1[2] - box2[2]) / (box1[2] + box2[2])
        cd = (c + d) * w2 * (-1)
        exponential_cost = exp(ab) * exp(cd)
        return exponential_cost

    def total_cost(self,old_box, new_box, iou_thresh=0.3, linear_thresh=10000, exp_thresh=0.5):
        iou_score = self.box_iou(old_box, new_box)
        linear_cost = self.sanchez_matilla(old_box, new_box)
        exponential_cost = self.yu(old_box, new_box)
        if (iou_score >= iou_thresh and linear_cost >= linear_thresh and exponential_cost >= exp_thresh):
            return iou_score
        else:
            return 0

    # This is the method for drawing the bounding boxes and ids on the image
    def id_to_color(self,idx):
        """
        Random function to convert an id to a color
        Do what you want here but keep numbers below 255
        """
        blue = idx * 5 % 256
        green = idx * 12 % 256
        red = idx * 23 % 256
        return (red, green, blue)


    def associate(self,old_boxes, new_boxes):
        """
        old_boxes will represent the former bounding boxes (at time 0)
        new_boxes will represent the new bounding boxes (at time 1)
        Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
        """
        if (len(new_boxes) == 0) and (len(old_boxes) == 0):
            return [], [], []
        elif (len(old_boxes) == 0):
            return [], new_boxes, []
        elif (len(new_boxes) == 0):
            return [], [], old_boxes

        # Define a new IOU Matrix nxm with old and new boxes
        iou_matrix = np.zeros((len(old_boxes), len(new_boxes)), dtype=np.float32)

        # Go through boxes and store the IOU value for each box
        # You can also use the more challenging cost but still use IOU as a reference for convenience (use as a filter only)
        for i, old_box in enumerate(old_boxes):
            for j, new_box in enumerate(new_boxes):
                iou_matrix[i][j] = self.total_cost(old_box, new_box)

        # Call for the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        # Create new unmatched lists for old and new boxes
        matches, unmatched_detections, unmatched_trackers = [], [], []

        # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched
        # Else: add the match
        for h in hungarian_matrix:
            if (iou_matrix[h[0], h[1]] < 0.3):
                unmatched_trackers.append(old_boxes[h[0]])
                unmatched_detections.append(new_boxes[h[1]])
            else:
                matches.append(h.reshape(1, 2))

        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        # Go through old boxes, if no matched detection, add it to the unmatched_old_boxes
        for t, trk in enumerate(old_boxes):
            if (t not in hungarian_matrix[:, 0]):
                unmatched_trackers.append(trk)

        # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
        for d, det in enumerate(new_boxes):
            if (d not in hungarian_matrix[:, 1]):
                unmatched_detections.append(det)

        return matches, unmatched_detections, unmatched_trackers

    # This is the method to convert the Dic list to the format required for tracking
    def trackConvertor(self,out_boxes):
        '''
        out_boxes : This is a list consisting of multiple dictionaries of bounding boxes one wants to track in a frame
        return : This method returns a list of bounding boxes in a frame
        '''
        finalboxes = []
        finalClasses = []
        for key in out_boxes.keys():
            # Take only the bounding box information from the boxes for the specific key
            out_box_boxes = out_boxes[key]
            # Append the boxes in the final boxes
            for box in out_box_boxes:
                finalboxes.append(box)
            # Extract the class information from the boxes
            #out_box_classes = list(out_boxes.keys())
            #for cls in out_box_classes:
            finalClasses.append(key)
        return finalboxes,finalClasses


    def Trackmain(self,input_image,out_boxes):
        """
        input_image ; This is the input frame for tracking the objects
        out_boxes : THis is the bounding boxes detected for the object in the frame as a dictionary containing classes and boxes
        """
        #global stored_obstacles
        #global idx

        # 1 — Run Obstacle Detection & Convert the Boxes
        final_image = copy.deepcopy(input_image)
        h, w, _ = final_image.shape
        #_, out_boxes, _, _ = inference(input_image)
        # Convert the boxes from a dictionary form to a list
        finalboxes, finalClasses = self.trackConvertor(out_boxes)

        #print("out_box_boxes,out_box_class",finalboxes,finalClasses)

        # print("----> New Detections: ", out_boxes)

        # Define the list we'll return:
        new_obstacles = []
        # Get the boxes from the old objects which are tracked
        old_obstacles = [obs.box for obs in self.stored_obstacles]  # Get the boxes from the stored obstacle class
        #print("Printing the old obstacles box",old_obstacles)
        # Create association between the new detections and old detections and get the matching boxes and detection
        matches, unmatched_detections, unmatched_tracks = self.associate(old_obstacles, finalboxes)  # Associate the obstacles

        print('matches',matches)
        print('unmatched_detections', unmatched_detections)
        print('unmatched_tracks', unmatched_tracks)

        #### To start cleaning up from this stage onwards

        # Matching
        for match in matches:
            obs = ObjectTracker(self.stored_obstacles[match[0]].idx, finalboxes[match[1]],final_image,self.stored_obstacles[match[0]].age +1)
            new_obstacles.append(obs)
            print("Obstacle ", obs.idx, " with box: ", obs.box, "has been matched with obstacle ", self.stored_obstacles[match[0]].box, "and now has age: ", obs.age)
            #print("Obstacle ", obs.idx, " with box: ", obs.box, "has been matched with obstacle ",self.stored_obstacles[match[0]].box)
        # New (Unmatched) Detections
        for new_obs in unmatched_detections:
            # Add the id, bounding box and the image as the ObjectTracker class
            obs = ObjectTracker(self.idx, new_obs,final_image)
            # Append it to the list
            new_obstacles.append(obs)
            self.idx += 1
            print("Obstacle ", obs.idx, " has been detected for the first time: ", obs.box)

        # Unmatched Tracking: NO RE-IDENTIFICATION FOR NOW
        for old_track in unmatched_tracks:
            i = old_obstacles.index(old_track)
            # print("Old Obstacles tracked: ", stored_obstacles[i].box)
            if i is not None:
                obs = self.stored_obstacles[i]
                obs.unmatched_age += 1
                new_obstacles.append(obs)
                print("Obstacle ", obs.idx, "is a long term obstacle unmatched ", obs.unmatched_age, "times.")

        # Draw the Boxes
        for i, obs in enumerate(new_obstacles):
            if obs.unmatched_age > self.MAX_UNMATCHED_AGE:
                new_obstacles.remove(obs)
            # the obs.box is in a dictionary format with the key as the class name and value as bounding boxes in a list of list
            if obs.age >= self.MIN_HIT_STREAK:
                # Store the image as the final image with bounding boxes
                obs.frame = input_image
                left, top, right, bottom = obs.box
                #print("left,top,right,bottom",left,top,right,bottom)
                cv2.rectangle(final_image, (left, top), (right, bottom), self.id_to_color(obs.idx * 10), thickness=7)
                final_image = cv2.putText(final_image, str(obs.idx), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          self.id_to_color(obs.idx * 10), thickness=4)


        self.stored_obstacles = new_obstacles

        # Saving all the existing obstacles in a dictionary to share and track
        allObstacles = self.obstacleExtractor()

        return final_image,allObstacles

    # This is the method to extract the various details from the self. stored obstacles
    def obstacleExtractor(self):
        # Initialise a list to store the obstacles
        allObstacles = []

        # Filter through each of the objects in the Tracker
        for i ,obj in enumerate(self.stored_obstacles):
            # Define a dictionary for storing the details
            trackDic = {}
            # Store the object id
            trackDic['objectID'] = i
            trackDic['trackID'] = obj.idx
            trackDic['box'] = obj.box
            trackDic['age'] = obj.age
            trackDic['img'] = obj.frame
            # Save the dictionary in the list
            allObstacles.append(trackDic)
        return allObstacles

