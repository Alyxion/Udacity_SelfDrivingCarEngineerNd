#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

# This script calibrates the camera using the chessboard images provided in the last project
# and trains a vehicle vs non-vehicle classifier which is later used to detect vehicles in a scene.
# This script dumps it's data in the files camera_calib.pkl and vehicle_classifier.pkl so these
# steps can be skipped by follow up scripts.

# Add reference to files of the 4th project
import sys
import os
rel_path = os.path.join(os.path.dirname(__file__), '..', '1_16_Project_4_Advanced_Lane_Lines')
sys.path.append(rel_path)

import AdvLaneCamera

from Vehicle_Trainer import Vehicle_Trainer
import matplotlib.pyplot as plt

# Calibrate camera
print("Calibrating camera...")
camera = AdvLaneCamera.AdvCamera()
calibration_path = "../1_16_Project_4_Advanced_Lane_Lines/camera_cal/calibration*.jpg"
camera.chessboard_calibrate_camera(search_path=calibration_path, show_images=3)
camera.save_to_pickle()
print("Done")
plt.show()

# Learn from vehicle vs non-vehicle pictures
print("Training vehicle classifier...")
trainer = Vehicle_Trainer()
trainer.train_vehicles('./../data/1_20_Object_Detection/classify/**/*.png')
trainer.save_to_disk()
print("Done")