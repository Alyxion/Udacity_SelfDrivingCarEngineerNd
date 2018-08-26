#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import sys
import os
rel_path = os.path.join(os.path.dirname(__file__), '..', '1_16_Project_4_Advanced_Lane_Lines')
sys.path.append(rel_path)

import AdvLaneVideoCreator

from moviepy.editor import VideoFileClip
import Hog_Me
import Vehicle_Trainer
import Object_Finder
import Object_Classifier
import Heater
import cv2
import numpy as np

class Vehicle_Video_Creator:
    """
    Manages the creation of the project specific outputs

    Properties:

    trainer: The Vehicle_Trainer, which already setup the Hoggit and the Object_Classifier
    finder: The object finder which detects the occurrences
    heater: The heater which helps finding most likely hits
    camera: The AdvCamera object to undistort the distorted input images
    add_lanes: Defines if the lanes from the previous project shall be added as well
    """

    def __init__(self, trainer, camera):
        self.trainer = trainer
        self.finder = Object_Finder.Object_Finder(trainer.classifier, trainer.hoggit)
        self.heater = Heater.Heater()
        self.camera = camera
        self.add_lanes = False

    def process_image(self, image):

        if self.add_lanes:
            warped, warped_cam, image = AdvLaneVideoCreator.lane_finder.find_lanes_using_window(image)
        else:
            image = self.camera.undistort(image)

        self.finder.find_objects_in_image(image, False)

        self.heater.add_to_history(self.finder.boundings)
        heat_image, color_heat = self.heater.get_heat(image)

        maxv = heat_image.max()
        if maxv == 0:
            maxv = 1.0
        heat_image *= 255 / maxv

        rgb_img = cv2.cvtColor(heat_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        small = cv2.resize(rgb_img, (128 * 2, 72 * 2))

        image = np.copy(image)

        image = self.heater.draw_labeled_bboxes(image)

        xoff = 1280-30-small.shape[1]
        yoff = 30
        image[yoff:yoff + small.shape[0], xoff:xoff + small.shape[1]] = small
        cv2.rectangle(image, (xoff, yoff), (xoff + small.shape[1], yoff + small.shape[0]), (255,255,255), 1)

        image = cv2.addWeighted(image, 1, color_heat, 0.6, 0)

        return image

    def process_video(self, video, file_name):
        white_clip = video.fl_image(self.process_image)
        white_clip.write_videofile(file_name, audio=False)