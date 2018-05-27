#####################################################################################################################
#                                                                                                                   #
# This file is part of the 4th project of Udacity's Self-Driving Car Engineer project Advanced Lane Finding         #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class AdvCamera:
    """
    The AdvCamera class is responsible for undistorting the input image with the help of a previously calculated
    camera matrix and distortion coefficients using OpenCV's calibrate camera function and a set of chessboard images.

    Attributes:
        mtx : The camera matrix
        dist : The distortion coefficients
    """
    def __init__(self):
        self.mtx = None
        self.dist = None

    def calibrate_camera(self, images):
        """
        Calibrates the camera using a set of images
        :param images: A list of images containing a chessboard with 3 inner column corners and 7 inner row corners
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        col_count = 3
        row_count = 7
        fig = plt.figure(figsize=(16, 32))

        index = 0

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                sp = fig.add_subplot(row_count, col_count, index + 1)
                plt.imshow(img)
                index += 1

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def chessboard_calibrate_camera(self):
        """
        Calibrates the camera using a set of images provided for the Advanced Lane Finding project
        """
        images = glob.glob('camera_cal/calibration*.jpg')

        self.calibrate_camera(images)

    def save_to_pickle(self):
        """
        Stores the calibration data to the file camera_calib.pkl
        """
        output = open('camera_calib.pkl', 'wb')
        pickle.dump(self.mtx, output)
        pickle.dump(self.dist, output)
        output.close()

    def load_from_pickle(self):
        """
        Loads the calibration data from the file camera_calib.pkl
        """
        camdata = open('camera_calib.pkl', 'rb')
        self.mtx = pickle.load(camdata)
        self.dist = pickle.load(camdata)
        camdata.close()

    def undistort(self, image):
        """
        Undistorts a camera image
        :param image: The original
        :return: The undistorted image
        """
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)