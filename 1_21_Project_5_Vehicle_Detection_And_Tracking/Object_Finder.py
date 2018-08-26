#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import Object_Classifier
import Hog_Me
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Object_Finder:
    """
    Finds all instances of an object previously trained by an Object_Classifier in an image.

    Properties:

    Classifier: The Object_Classifier to be used to find occurrences of an object.
    box_size: The size of the object / search window size
    eff_box_size: The size of the scaled box size. Applied by set_scaling.
    boundings: The boundings of the recently detected objects
    single_hog: Defines if HOG shall just be created once for the whole image. Otherwise it will be created for each
    window (far slower).
    hogger: The image to hog converter to used.
    resized_image: A backup of the resized image
    scan_regions: Defines the regions to be scanned and by which factor the search window shall be magnified in this
    region. (The closer to the observer the higher the factor should be)
    original: The currently selected original image
    """

    def __init__(self, classifier, hogger):
        """
        Constructor
        :param classifier: The classifier of class Object_Classifier to be used to detect the object
        :param hogger: The image to hog convert of class Hoggit
        """
        self.classifier = classifier
        self.box_size = classifier.box_size
        self.boundings = []
        self.single_hog = True
        self.hogger= hogger
        self.resized_image = None
        # Setup magnifications for single scan regions
        self.scan_regions = [(1.0, (0, 380, 1280, 500)),
                        (1.25, (0, 380, 1280, 550)),
                        (1.5, (0, 380, 1280, 550)),
                        (1.75, (0, 380, 1280, 550)),
                        (2.0, (0, 380, 1280, 650))]

    def select(self, image):
        """
        Selects a new image and clears the old detections
        :param image: The new image to be used
        """
        self.original = image
        self.boundings = []

    def set_scaling(self, scaling):
        """
        Sets the scaling factor to be used for new detections
        :param scaling: The factor by which the search window shall be enlarged. (>1.0 for objects close to the eye)
        """
        self.scaling = scaling
        self.eff_box_size = int(self.box_size*self.scaling+0.5)

    def get_resized_sub_sample(self, off_x, off_y):
        """
        Returns the sub region of the currenty selected original image
        :param off_x: The x offset
        :param off_y: The y offset
        :return: An image resized from the current effective box size to the original box size
        """
        sub_sample = self.original[off_y:off_y + self.eff_box_size, off_x:off_x + self.eff_box_size, :]
        new_size = (self.box_size, self.box_size)
        return cv2.resize(sub_sample, new_size)

    def find_instances_in_features(self, features, region):
        """
        Finds the instances of an object in the currently selected image.
        :param features: The feature list of the sub region of the main image
        :param region: Defines the region of the main image which is represents by features
        """
        for current_window in self.hogger.hog_scan(self.box_size, self.box_size):
            if self.classifier.classify_features(current_window["features"])==1.0:
                off_x = current_window["x"]
                off_y = current_window["y"]
                trans_off_x = int(off_x * self.scaling) + region[0]
                trans_off_y = int(off_y * self.scaling) + region[1]

                cv2.rectangle(self.resized_image, (off_x, off_y), (off_x + self.box_size, off_y + self.box_size),
                              color=(255, 255, 255), thickness=2)
                cv2.rectangle(self.image, (trans_off_x, trans_off_y), (trans_off_x + self.eff_box_size, trans_off_y + self.eff_box_size),
                              color=(255, 255, 255), thickness=2)
                self.boundings.append(((trans_off_x, trans_off_y), (trans_off_x + self.eff_box_size, trans_off_y + self.eff_box_size)))

    def find_instances(self, image, region, overlap):
        """
        Finds all instances of the object the classifier has been trained on in given image."

        The results are appened to property boundings.

        :param image: The image to search in
        :param region: The sub region to search within
        :param overlap: The overlap in percent. Only required if single_hog is det to False.
        :return: The original with highlighted detection regions and optionally the resized sub image
        """
        self.image = np.copy(image)

        self.eff_step_size = int((1.0-overlap)*self.eff_box_size)

        y_steps = (region[3]-region[1])//self.eff_step_size
        x_steps = (region[2]-region[0])//self.eff_step_size

        if region[0]+(x_steps-1)*self.eff_step_size+self.eff_box_size>region[2]:
            x_steps -= 1
        if region[1]+(y_steps-1)*self.eff_step_size+self.eff_box_size>region[3]:
            y_steps -= 1

        if self.single_hog:
            self.resized_image = image[region[1]:region[3],region[0]:region[2],:]
            self.resized_image = cv2.resize(self.resized_image, (int(self.resized_image.shape[1]/self.scaling), int(self.resized_image.shape[0]/self.scaling)))
            features, img = self.hogger.hog_image(self.resized_image, visualize=False, feature_vector=False)
            features = np.array(features)
            self.find_instances_in_features(features, region)
            return self.image, self.resized_image
        else:
            for row in range(y_steps):
                off_y = region[1] + row * self.eff_step_size
                for col in range(x_steps):
                    off_x = region[0]+col * self.eff_step_size
                    sub_sample = self.get_resized_sub_sample(off_x, off_y)
                    pred = self.classifier.classify(sub_sample)
                    if(pred==1.0):
                        cv2.rectangle(self.image, (off_x, off_y), (off_x+self.eff_box_size, off_y+self.eff_box_size), color=(255,255,255), thickness=2)
                        self.boundings.append(((off_x, off_y), (off_x+self.eff_box_size, off_y+self.eff_box_size)))

            return self.image, None

    def find_objects_in_image(self, image, visualize):
        """
        Finds all instances detected by the classifier in the image provided.

        :param image: The image to search in
        :param visualize: Defines if the single region results shall be visualized using plotly.
        """
        self.select(image)

        for scan_region in self.scan_regions:
            self.set_scaling(scan_region[0])
            full, small = self.find_instances(image, scan_region[1], 0.5)
            if (visualize):
                fig = plt.figure(figsize=(30, 20))
                plt.imshow(small)
                plt.title("Scan region {} using scaling {}".format(scan_region[1], scan_region[0]))
                plt.show()