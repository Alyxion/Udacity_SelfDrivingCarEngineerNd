#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import numpy as np
from skimage.feature import hog
import cv2

class Hoggit:
    """
    Huggers gonna hug - Hoggers gonna hog ;-)

    The hoggit class is responsible for converting an image into a so called "History of Oriented Gradients", short
    HOG by detecting how intensive specific gradient directions (for example vertically, horizontally or between)
    are in a specific region of like 8x8 pixels.

    By collecting many positive and negative HOGs, lets say from bananas and non-bananas and feeding the histogram
    into a classifier such as SVM or a neural network you may then predict if a new picture is likely a banana as well
    or not.

    Properties:

    orientations: The number of orientations (histogram buckets). 9 is a common value to detect gradients in 20 degree steps.
    pix_per_cell: Defines the width and height of a histogram cell
    cells_per_block: Defines the width and height of a block. A single block is just a collection of (shared) cell
    results. The cells though don't contain precisely the same values because each block is normalized individually.
    features: The most recent detected features.
    vis_image: The most recent painted hog image
    channel: The channel to be used for creating the hog. "ALL" if all channels shall be considered.
    target_format: The target format into which the RGB image shall be considere. A CV2 conversion enum is required such
    as cv2.COLOR_RGB2YUV.
    """

    def __init__(self, channel="ALL", orientations = 9, pix_per_cell = 8, cells_per_block = 2, target_format = cv2.COLOR_RGB2YUV):
        """
        Constructor
        :param channel: The channel to be considered or "ALL" for all channels. "ALL" by default
        :param orientations: The number of orientations buckets. 9 by default
        :param pix_per_cell: The width and height of a cell in pixels
        :param cells_per_block: The width and height of a block in cells
        :param target_format: The CV2 conversion enum for the RGB original. COLOR_RGB2YUV by default.
        """
        self.orientations = orientations
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.features = None
        self.vis_image = None
        self.channel = channel
        self.target_format = target_format

    def hog_image(self, img, visualize = True, feature_vector = True):
        """
        Converts an image into a list of blocks of oriented gradients.
        :param img: The input image
        :param visualize: Defines if the HOGs shall be visualized
        :param feature_vector: Defines if the result shall be flattened. (1D if True, multi-dimensional if not)
        :return: The feature vector and the visualization. None as second result of visualize is set to False.
        """
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.cells_x = self.width // self.pix_per_cell
        self.cells_y = self.height // self.pix_per_cell

        self.x_blocks = self.cells_x - self.cells_per_block + 1
        self.y_blocks = self.cells_y - self.cells_per_block + 1
        self.total_blocks = self.x_blocks * self.y_blocks
        self.total_features = self.total_blocks * self.cells_per_block * self.cells_per_block * self.orientations

        img= cv2.cvtColor(img, self.target_format)

        if self.channel == 'ALL':
            features = []
            for channel in range(img.shape[2]):
                features.extend(hog(img[:,:,channel], orientations=self.orientations,
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cells_per_block, self.cells_per_block),
                           block_norm='L2-Hys',
                           transform_sqrt=False,
                           visualise=visualize, feature_vector=feature_vector))
            hog_img = None
        else:
            if visualize:
                features, hog_img = hog(img[:, :, self.channel], orientations=self.orientations,
                                        pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                        cells_per_block=(self.cells_per_block, self.cells_per_block),
                                        block_norm='L2-Hys',
                                        transform_sqrt=False,
                                        visualise=visualize, feature_vector=feature_vector)
            else:
                features = hog(img[:, :, self.channel], orientations=self.orientations,
                               pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                               cells_per_block=(self.cells_per_block, self.cells_per_block),
                               block_norm='L2-Hys',
                               transform_sqrt=False,
                               visualise=visualize, feature_vector=feature_vector)
                hog_img = None

        self.vis_image = hog_img
        self.features = np.array(features)

        return features, hog_img

    def hog_scan(self, window_width, window_height, step_factor = 1):
        """
        An enumerator which scans the last hog result using the given window size.
        :param window_width: The width of the window in pixels
        :param window_height: The height of the window in pixels
        :param step_factor: The block stepping (1 by default)
        :return: A dictionary containing the "features" in the current region, and the top left "x" and "y" coordinate.
        """
        if self.channel == "ALL":
            all_channels = True
            offset_per_channel = self.features.shape[0]//3
        else:
            all_channels = False
            offset_per_channel = 0

        cells_per_block = self.cells_per_block
        pix_per_cell = self.pix_per_cell

        block_size = cells_per_block*cells_per_block*pix_per_cell
        x_range = window_width//pix_per_cell - cells_per_block+ 1
        y_range = window_height//pix_per_cell - cells_per_block+ 1

        x_steps = self.features.shape[1]-x_range+1
        y_steps = offset_per_channel-x_range+1

        for y_off in range(y_steps):
            for x_off in range(x_steps):
                xpos = x_off*step_factor
                ypos = y_off*step_factor

                hog_feat1 = self.features[ypos:ypos+y_range, xpos:xpos+x_range].ravel()

                if all_channels:
                    hog_feat2 = self.features[offset_per_channel+ypos:offset_per_channel+ypos+y_range, xpos:xpos+x_range].ravel()
                    hog_feat3 = self.features[2*offset_per_channel+ypos:2*offset_per_channel+ypos+y_range, xpos:xpos+x_range].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog_feat1

                off_x = int(xpos * pix_per_cell)
                off_y = int(ypos * pix_per_cell)

                yield {"features": hog_features, "x": off_x, "y": off_y}
