import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

def calibrate_camera(images):
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

    # Step throuagh the list and search for chessboard corners
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

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def chessboard_calibrate_camera():
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    mtx, dist = calibrate_camera(images)

    output = open('camera_calib.pkl', 'wb')
    pickle.dump(mtx, output)
    pickle.dump(dist, output)
    output.close()

    plt.show()

chessboard_calibrate_camera()