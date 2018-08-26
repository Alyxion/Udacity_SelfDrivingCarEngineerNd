#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import cv2
import numpy as np
from scipy.ndimage.measurements import label

class Heater:
    """
    The Heater class is responsible for computing the intensity of the likeliness of an object being located in
    a specific region of the image.

    To do so it creates a heatmap of several frames and afterwards cuts all activations below a minimum threshold.

    Afterwards scikit is used to build regions around single instances. These instances are then filtered if
    they are too small for a given position on the screen (and so relative to the car)

    Properties:

    history: An array of rectangle arrays in which the whole history of possible detections is recorded
    max_remembering: Defines how many frames the heatmap looks into the past to consider previous frame's detections
    base_scale: Defines a scaling factor by which all detections are scaled. This should be higher than max_remebering.
    labels: The detected, potential object regions
    min_size_default: The minimum size of a detected object which shall still be considered
    size_threshholds: Defines the minimium size of an object required if it's bottom is below given y coordinate.
    threshhold: Defines the minimum activation of a single pixel so it will be kept.
    The closer an object is to the observer the larger it should be.


    """

    def __init__(self):
        """
        Constructor
        """
        self.history = []
        self.max_remembering = 5
        self.base_scale = 5
        self.labels = []
        self.min_size_default = 40
        self.size_threshholds = [(600,150), (550,110), (500, 80), (480,60), (450,40)]
        self.threshhold = self.max_remembering*self.base_scale*5
        self.last_unthreshed = None

    def add_to_history(self, rectangles):
        """
        Adds the current list of rectangles to the history
        :param rectangles: The list of rectangles
        """
        self.history.append(rectangles)

    def add_heat(self, heatmap):
        """"
        Adds heat to a heapmap using the current list of bounding boxes in the history
        """

        look_back = self.max_remembering
        hist_start = len(self.history)-1
        if hist_start-look_back+1<0:
            look_back = hist_start+1

        # For all relevant entires in history: Increase heat
        for hist_index in range(hist_start-look_back+1,hist_start+1):
            for box in self.history[hist_index]:
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += int(self.base_scale*self.max_remembering/look_back)

        return heatmap

    def draw_labeled_bboxes(self, img):
        """
        Draws a bounding box around every detected object
        :param img: The target image
        :return: The target image
        """
        for bbox in self.labels:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 2)

        return img

    def get_heat(self, image):
        """
        Returns the current heat maps
        :param image: The reference image
        :return: An image containing the intensity in gray and an image containing the intensity as RGB image.
        """

        # Calculate heat
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = self.add_heat(heat)
        color_heat = np.zeros_like(image[:, :, :]).astype(np.float)

        # Remove all values which are too small
        self.last_unthreshed = np.copy(heat)
        heat[heat <= self.threshhold] = 0
        color_heat[heat!=0] = (170,0,0)
        color_heat = color_heat.astype(np.uint8)

        invalid_boxes = []

        # Use scikit to collect instances
        new_labels = label(heat)
        self.labels = []

        # For all potentially detected instances...
        for car_number in range(1, new_labels[1] + 1):
            nonzero = (new_labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            width = bbox[1][0] - bbox[0][0]
            height= bbox[1][1] - bbox[0][1]

            min_size = self.min_size_default

            # Get position dependent minimum size
            for tr_region in self.size_threshholds:
                if bbox[1][1]>tr_region[0]:
                    min_size = tr_region[1]
                    break

            # If rectangle is large enough for it' region...
            if (width > min_size) and (height > min_size):
                self.labels.append(bbox)
            else:
                invalid_boxes.append(bbox)

        for box in invalid_boxes:
            color_heat[box[0][1]:box[1][1]+1, box[0][0]:box[1][0]+1] = 0

        return heat, color_heat
