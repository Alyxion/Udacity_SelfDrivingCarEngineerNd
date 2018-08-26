#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

# With help of this file you can compile project video by just running it.
# If you did not do so yet you need to call the script Train_Classifier once to calibrate the camera and retrain
# the classifier.

# Add reference to files from previous project
import sys
import os
rel_path = os.path.join(os.path.dirname(__file__), '..', '1_16_Project_4_Advanced_Lane_Lines')
sys.path.append(rel_path)

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import Vehicle_Video_Creator
import AdvLaneCamera
import AdvLaneVideoCreator
import Vehicle_Trainer
import Hog_Me
import Object_Finder
import Object_Classifier
import Heater

# Load camera calibration
camera =  AdvLaneCamera.AdvCamera()
camera.load_from_pickle()

# Load vehicle detection neural network
trainer = Vehicle_Trainer.Vehicle_Trainer()

do_train = False
if do_train:
    print("Retraining classifier")
    trainer.classifier.use_svm = False
    trainer.train_vehicles('./../data/1_20_Object_Detection/classify/**/*.png', visualize=False)
    trainer.save_to_disk()
else:
    trainer.load_from_disk()

# Create video
video_creator = Vehicle_Video_Creator.Vehicle_Video_Creator(trainer=trainer, camera=camera)

output_video_name = 'test_videos_output/result.mp4'
input_video_name = "project_video.mp4"

input_clip = VideoFileClip(input_video_name)
# sub_clip = input_clip.subclip(48,51)
# sub_clip = input_clip.subclip(25,30)
# sub_clip = input_clip.subclip(0,2)
sub_clip = input_clip

video_creator.add_lanes = True
video_creator.process_video(sub_clip, output_video_name)