#####################################################################################################################
#                                                                                                                   #
# This file is part of the 4th project of Udacity's Self-Driving Car Engineer project Advanced Lane Finding         #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import numpy as np
import cv2

class LanePerspectiveTransform:
    """
    The LanePerspectiveTransform class helps transforming the perspective camera image to a top down view for
    an easier evaluation as well as backtransforming found lane lines from topdown coordinates back to the
    perspective view.

    Properties:

    tmx: The transformation matrix from perspectival view to top view
    retmx: The transformation matrix from top view to perspectival view
    """

    def __init__(self, calibration_image):
        img_size = (calibration_image.shape[1], calibration_image.shape[0])

        # defines the perspective of the camera image by defining points near the front of the car and close the the
        # center of the image
        relation_factor = 15.5/1.92
        front_perspective_div = 3.0
        back_perspective_div = front_perspective_div*relation_factor
        front_perspective_y_perc = 0.98
        back_perspective_y_perc = 0.63
        margin_factor = 6

        # trapez points in order bottom left, top left, top right, bottom right (front, back, back, front)
        self.org_src = [(int(img_size[0]/2-img_size[0]/front_perspective_div), int(img_size[1]*front_perspective_y_perc)),
               (int(img_size[0]/2-img_size[0]/back_perspective_div), int(img_size[1]*back_perspective_y_perc)),
               (int(img_size[0]/2+img_size[0]/back_perspective_div), int(img_size[1]*back_perspective_y_perc)),
               (int(img_size[0]/2+img_size[0]/front_perspective_div), int(img_size[1]*front_perspective_y_perc))]

        org_dst = [(img_size[0]//margin_factor, img_size[1]),
               (img_size[0]//margin_factor, 0),
               (img_size[0]-img_size[0]//margin_factor, 0),
               (img_size[0]-img_size[0]//margin_factor, img_size[1]),
               ]

        src = np.array(self.org_src, dtype=np.float32)
        dst = np.array(org_dst, dtype=np.float32)

        self.tmx = cv2.getPerspectiveTransform(np.array(src), np.array(dst))
        self.retmx = cv2.getPerspectiveTransform(np.array(dst), np.array(src))

    def transform_perspective_top(self, image):
        """
        Transforms a perspective view of a street to a top down view
        :param image: The perspective camera image
        :return: The top view camera image
        """
        width = image.shape[1]
        height = image.shape[0]
        return cv2.warpPerspective(image, self.tmx, (width, height))

    def transform_top_perspective(self, image):
        """
        Transforms a top view camera image back to a perspective image
        :param image:
        :return: The perspective view image
        """
        width = image.shape[1]
        height = image.shape[0]
        return cv2.warpPerspective(image, self.retmx, (width, height))

    def transform_top_perspective_coords(self, coordinates):
        """
        Transforms a list of coordinates from a top down view to a perspective view
        :param coordinates: The input coordinates
        :return: The perspective coordinates
        """
        return cv2.perspectiveTransform(coordinates, self.retmx)
