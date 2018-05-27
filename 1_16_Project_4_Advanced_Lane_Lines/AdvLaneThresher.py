#####################################################################################################################
#                                                                                                                   #
# This file is part of the 4th project of Udacity's Self-Driving Car Engineer project Advanced Lane Finding         #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import cv2
import numpy as np

class AdvLaneThresher:
    """
    The AdvLaneThresher is responsible for finding likely lane lines using an hls filter and a sobel filter
    """

    def __init__(self):
        """
        Constructor
        """
        self.hls_thresh = (95, 255)
        self.sobal_mag_thresh = (10, 255)
        self.sobel_mag_kernel = 3
        self.sobal_dir_thresh = (0.7, 1.3)
        self.sobel_dir_kernel = 13

    def hls_threshold_mask(self, image):
        """
        Highlights pixels with a quite high saturation
        :param image: The original image
        :return: A binary mask flagging all potential lane pixels
        """
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        s = hls[:, :, 2]

        binary_output = np.zeros_like(s)
        binary_output[(s >= self.hls_thresh[0]) & (s <= self.hls_thresh[1])] = 1

        return s, binary_output

    def sobel_mag_mask(self, image):
        """
        Highlights pixels which are very likely part of an edge
        :param image: The original image
        :return: A binary mask flagging all potential lane pixels
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.sobel_mag_kernel)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.sobel_mag_kernel)
        abs_sobelxy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        eight_bit = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
        # 5) Create a binary mask where mag thresholds are met
        # 6) Return this mask as your binary_output image
        binary_output = np.zeros_like(abs_sobelxy)
        binary_output[(eight_bit >= self.sobal_mag_thresh[0]) & (eight_bit <= self.sobal_mag_thresh[1])] = 1

        return eight_bit, binary_output

    def sobel_dir_mask(self, image):
        """
        Highlights pixels which are likely part of a mostly vertical ine
        :param image: The original image
        :return: The binary mask highlighting likely lane pixels
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.sobel_dir_kernel)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.sobel_dir_kernel)
        abs_x = np.abs(sobel_x)
        abs_y = np.abs(sobel_y)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        sob_dir = np.arctan2(abs_y, abs_x)
        # 5) Create a binary mask where direction thresholds are met
        # 6) Return this mask as your binary_output image
        binary_output = np.zeros_like(sob_dir)
        binary_output[(sob_dir >= self.sobal_dir_thresh[0]) & (sob_dir <= self.sobal_dir_thresh[1])] = 1

        return sob_dir, binary_output

    def create_binary_mask(self, image):
        """
        Highlights pixels which are likely part of the lane lines using several filters
        :param image: The original image
        :return: A binary mask containing pixels which are likely part of the lanes
        """
        s, hls_thresh = self.hls_threshold_mask(image)
        # mag, sobel_mag = self.sobel_mag_mask(image)
        dir, sobel_dir = self.sobel_dir_mask(image)

        binary_output = np.zeros_like(sobel_dir)
        binary_output[(hls_thresh==1) & (sobel_dir==1)  ] = 1

        return binary_output