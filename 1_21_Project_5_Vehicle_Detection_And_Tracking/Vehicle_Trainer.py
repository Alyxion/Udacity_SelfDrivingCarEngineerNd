#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import cv2
import Hog_Me
import Object_Classifier
import glob
import pickle
import matplotlib.pyplot as plt

class Vehicle_Trainer:
    """
    Trains the project's vehicle classifier and setups the global image to hov converter
    """

    def __init__(self):
        # HOG configuration
        self.hog_pix_per_cell = 16
        self.hog_orientations = 12
        self.hog_cells_per_block = 2
        self.hog_channels = "ALL"
        self.hog_format = cv2.COLOR_RGB2YUV
        self.hoggit = Hog_Me.Hoggit(channel=self.hog_channels, orientations=self.hog_orientations, pix_per_cell=self.hog_pix_per_cell,
                    cells_per_block=self.hog_cells_per_block, target_format=self.hog_format)

        # Setup classifier
        self.classifier = Object_Classifier.Object_Classifier(self.hoggit)

    def train_vehicles(self, search_path, visualize = True):
        """
        Trains the classifier using the images provided in search_path
        :param search_path: The search path including the scan mask
        :param visualize: Defines it the result views shall be visualized
        """
        images = glob.glob(search_path, recursive=True)

        cars = []
        notcars = []
        for image in images:
            if 'non-vehicles' in image:
                notcars.append(image)
            else:
                cars.append(image)

        sample_size = 8000
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

        self.classifier.train(cars, notcars)

        img = cv2.imread(cars[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prd = self.classifier.classify(img);

        if visualize:
            plt.imshow(img)
            plt.title("Vehicle = {}%".format(prd[0]*100))
            plt.show()

        print("Veryfing vehicle. Should return one: {}".format(prd[0]))

        img = cv2.imread(notcars[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prd = self.classifier.classify(img);

        if visualize:
            plt.imshow(img)
            plt.title("No vehicle = {}%".format((1.0-prd[0])*100))
            plt.show()

        print("Veryfing non-vehicle. Should return zero: {}".format(prd[0]))

    def save_to_disk(self, file_name = "vehicle_classifier.pkl"):
        """
        Stores the classifier's data to disk
        :param file_name: The target file name
        :return:
        """
        self.classifier.save_to_disk(file_name)

    def load_from_disk(self, file_name = "vehicle_classifier.pkl"):
        """
        Loads the classifier's data from disk
        :param file_name: The source file name
        :return:
        """
        self.classifier.load_from_disk(file_name)
