#####################################################################################################################
#                                                                                                                   #
# This file is part of the 4th project of Udacity's Self-Driving Car Engineer Nanodegree - Advanced Lane Finding    #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import cv2
import AdvLaneCamera

class AdvLaneHelper:
    """
    Just a little dirty helper
    """
    def __init__(self, camera):
        """
        Initializes the helper
        :param camera: The AdvLaneCamera object for undistorting loaded images
        """
        self.camera = camera

    def load_and_undistort(self, filename):
        """
        Loads an image from disk, undistorts it and converts it to RGB space
        :param filename:  The name of the image file to be loaded
        :return: The undistorted image
        """
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.camera.undistort(img)
        return img

    def get_example_images(self):
        """
        Returns a list of exapmle images
        :return: An array of example images of street situations
        """
        return ['test_images/test1.jpg', 'test_images/test2.jpg', 'test_images/test3.jpg',
         'test_images/test4.jpg', 'test_images/test5.jpg', 'test_images/test6.jpg',
         'test_images/straight_lines1.jpg', 'test_images/straight_lines2.jpg']