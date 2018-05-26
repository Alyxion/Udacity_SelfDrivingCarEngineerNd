import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import ntpath
import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML

class LaneFinder:
    def __init__(self):
        self.last_base_left = None
        self.last_base_right = None
        self.hist_minimum_thresh = 30
        self.hist_perf_thresh = 60
        self.lane_size = None
        self.perfect_pix_win = 140

    def find_lanes_using_window(self, img):
        sp = fig.add_subplot(row_count, col_count, index + 1)
        img = cv2.undistort(img, mtx, dist, None, mtx)
        binary_warped = create_binary_mask(img)
        binary_warped = cv2.warpPerspective(binary_warped, tmx, (width, height))

        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if histogram[leftx_base] <= self.hist_minimum_thresh and self.last_base_left is not None:
            leftx_base = self.last_base_left
        else:
            self.last_base_left = leftx_base

        if histogram[rightx_base] <= self.hist_minimum_thresh and self.last_base_right is not None:
            rightx_base = self.last_base_right
        else:
            self.last_base_right = rightx_base

        # if we can right now perfectly detect both lanes recalibrate the lane size
        if histogram[leftx_base] >= self.hist_perf_thresh and histogram[rightx_base] >= self.hist_perf_thresh:
            self.lane_size = rightx_base - leftx_base

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        dominant_l = False
        dominant_r = False

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            color_l = (0, 255, 0)
            color_r = (0, 255, 0)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            if len(good_left_inds) > self.perfect_pix_win:
                dominant_l = True
            else:
                dominant_l = False

            if len(good_right_inds) > self.perfect_pix_win:
                dominant_r = True
            else:
                dominant_r = False

            if dominant_l:
                color_l = (255, 0, 0)
            if dominant_r:
                color_r = (255, 0, 0)

                # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          color_l, 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          color_r, 2)


            # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = None
        right_fit = None

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        if leftx.shape[0] != 0 and lefty.shape != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        if rightx.shape[0] != 0 and righty.shape != 0:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        out_img = out_img.astype(np.ubyte)

        return out_img