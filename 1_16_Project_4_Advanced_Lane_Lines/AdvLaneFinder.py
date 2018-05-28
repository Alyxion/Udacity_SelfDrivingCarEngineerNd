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
import AdvLaneCamera
import AdvLaneHelper
import AdvLanePerspectiveTransform

class LaneFinder:
    """
    The LaneFinder class tries to detect lanes of a still image or video and returns a list of images into which
    it renders the detected lane markers.

    The only required function of this class is find_lanes_using_window which returns 3 images, a raw top view,
    a photo top view and the combined, perpsective view.
    """

    def __init__(self, camera, transform, thresher):
        """
        Constructor
        :param camera: A previously initialized AdvCamera object to correct camera distortion
        :param transform: A LanePerspectiveTransform object to transform from persective to top view and back
        :param thresher: An AdvLaneThresher object, applies several filters to detect the lanes in an image
        """
        # Helper classes
        # The camera distortion helper class
        self.camera = camera
        # The perspective distortion helper class
        self.transform = transform
        # The edge detector
        self.thresher = thresher

        # The current left lane start
        self.last_base_left = None
        # The current right lane start
        self.last_base_right = None
        # The minimum intensity in the histogram so a detected peak could be a lane
        self.hist_minimum_thresh = 30
        # The threshold in the histogram from where on there has very likely a lane been found
        self.hist_perf_thresh = 60
        # The current lane size estimation
        self.lane_size = None
        # Minimum number of flagged pixels to be found to move a window
        self.minpix = 50
        # Minimum number of flagged pixels for a sure match
        self.perfect_pix_win = 80


        # Defines when the distance between two lanes is likely errornerous
        self.valid_lane_width_error = 700
        # Defines the minimum width between two lanes
        self.valid_lane_width_min = 800
        # Defines the maximum width between two lanes
        self.valid_lane_width_max = 920

        # Input image
        self.original = None
        # Output images
        # The masked top view
        self.cur_mask = None
        # The top view photo
        self.out_transformed = None
        # The masked top view image containing markers
        self.out_img = None
        # The perspective, undistorted image
        self.out_perspective = None

        # The number of windows used for lane detection
        self.nwindows = 9
        # Each window's height
        self.window_height = None
        # The width of the windows +/- margin
        self.window_margin = 100

        # Defines if the output shall be interpolated over a given amount of frames, <=1 = not
        self.interpolation = 10
        # History of known left and right lane positions
        self.left_fit_history = []
        self.right_fit_history = []

        # Remember curvature
        self.curve_rad = None
        # Offset to road center
        self.offset_to_center = None

    def clear_history(self):
        """
        Clears caches such as frame interpolation
        """
        self.left_fit_history = []
        self.right_fit_history = []
        self.curve_rad = None


    def transform_original(self, image):
        """
        Receives the original image, undistorts it and backups it in a perspective and top view
        :param image: The original image
        """
        img = self.camera.undistort(image)
        self.original = img
        self.out_perspective = np.copy(img)
        warped = self.transform.transform_perspective_top(img)
        self.out_transformed = warped

    def select_mask_and_transform(self, image):
        """
        Creates a binary mask in a top view of the original image in which potential lane positions are highlighted
        :param image: The original image
        """
        img = self.camera.undistort(image)
        binary_warped = self.thresher.create_binary_mask(img)
        binary_warped = self.transform.transform_perspective_top(binary_warped)
        self.cur_mask = binary_warped
        self.window_height = np.int(self.cur_mask.shape[0] // self.nwindows)

    def find_dominant_lane(self):
        """
        After all windows positions have been computed this function evaluates if either the left or the right lane
        have a far more sure detection than the other one and if so sychronizes them so the left becomes a parallel
        version of the right or the other way round
        """
        self.left_super_dominant = False
        self.right_super_dominant = False

        self.detection_errors = 0
        self.max_detection_errors = 2

        # Verify the likeliness that our detected windows may be real or erroneous
        for window in range(self.nwindows):
            if self.windows_right_tl[window][0] - self.windows_left_tl[window][0] < self.valid_lane_width_error:
                self.detection_errors += 1

        # Are the locations of the left lane windows far more likely than the right ones or was there even no right lane found at all?
        if ((self.dominant_left_count>5 and self.dominant_right_count<4) or (self.dominant_left_count>7 and self.dominant_right_count<5)  or (self.dominant_right_count<2) or (self.detection_errors>=self.max_detection_errors and self.dominant_left_count>self.dominant_right_count)) and self.lane_size!=None:
            self.windows_right_tl.clear()
            self.windows_right_br.clear()

            left_super_dominant = True

            for window in range(self.nwindows):
                cur_tl = self.windows_left_tl[window]
                cur_br = self.windows_left_br[window]

                self.windows_right_tl.append((cur_tl[0]+self.lane_size, cur_tl[1]))
                self.windows_right_br.append((cur_br[0]+self.lane_size, cur_br[1]))

        # Do the same for the right lane
        if ((self.dominant_right_count>5 and self.dominant_left_count<4) or (self.dominant_right_count>7 and self.dominant_left_count<5) or (self.dominant_left_count<2) or (self.detection_errors>=self.max_detection_errors and self.dominant_right_count>self.dominant_left_count)) and self.lane_size!=None:
            self.windows_left_tl.clear()
            self.windows_left_br.clear()

            right_super_dominant = True

            for window in range(self.nwindows):
                cur_tl = self.windows_right_tl[window]
                cur_br = self.windows_right_br[window]

                self.windows_left_tl.append((cur_tl[0]-self.lane_size, cur_tl[1]))
                self.windows_left_br.append((cur_br[0]-self.lane_size, cur_br[1]))

    def find_windows(self):
        """
        Tries to find all locations of lanes
        The algorithm is starting at an initial position at the bottom and works itself upwards to the top of the image,
        so from the car's position towards the sky
        """
        # Step through the windows one by one

        self.windows_left_tl = []
        self.windows_left_br = []
        dominant_l_list = []
        self.dominant_left_count = 0
        self.windows_right_tl = []
        self.windows_right_br = []
        dominant_r_list = []
        self.dominant_right_count = 0
        self.window_colors_l = []
        self.window_colors_r = []

        # for all window locations, starting at the bottom of the image (near the car) and moving to the top (far away)
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.cur_mask.shape[0] - (window + 1) * self.window_height
            win_y_high = self.cur_mask.shape[0] - window * self.window_height
            win_xleft_low = self.leftx_current - self.window_margin
            win_xleft_high = self.leftx_current + self.window_margin
            win_xright_low = self.rightx_current - self.window_margin
            win_xright_high = self.rightx_current + self.window_margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                               (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                self.leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                self.rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

            win_xleft_low = self.leftx_current - self.window_margin
            win_xleft_high = self.leftx_current + self.window_margin
            win_xright_low = self.rightx_current - self.window_margin
            win_xright_high = self.rightx_current + self.window_margin

            # remember the likeliness of each detected window by the amount of successfully detected flagged pixels
            dominant_l = False
            dominant_r = False

            if len(good_left_inds) > self.perfect_pix_win:
                dominant_l = True
                self.dominant_left_count += 1

            if len(good_right_inds) > self.perfect_pix_win:
                dominant_r = True
                self.dominant_right_count += 1

            dominant_l_list.append(dominant_l)
            dominant_r_list.append(dominant_r)

            self.windows_left_tl.append((win_xleft_low, win_y_low))
            self.windows_left_br.append((win_xleft_high, win_y_high))
            self.windows_right_tl.append((win_xright_low, win_y_low))
            self.windows_right_br.append((win_xright_high, win_y_high))

        self.find_dominant_lane()

        for window in range(self.nwindows):

            if dominant_l_list[window]:
                color_l = (255, 0, 0)
            elif self.right_super_dominant:
                if self.detection_errors>=self.max_detection_errors:
                    color_l = (0, 0, 255)
                else:
                    color_l = (255, 255, 0)
            else:
                color_l = (0, 255, 0)

            if dominant_r_list[window]:
                color_r = (255, 0, 0)
            elif self.left_super_dominant:
                if self.detection_errors>=self.max_detection_errors:
                    color_r = (0, 0, 255)
                else:
                    color_r = (255, 255, 0)
            else:
                color_r = (0, 255, 0)

            self.window_colors_l.append(color_l)
            self.window_colors_r.append(color_r)

        self.paint_windows(self.out_img)
        self.paint_windows(self.out_transformed)

    def paint_windows(self, target):
        """
        Draws the detected windows
        :param target: The target image
        """
        for window in range(self.nwindows):

            # Draw the windows on the visualization image
            cv2.rectangle(target, self.windows_left_tl[window], self.windows_left_br[window],
                          self.window_colors_l[window], 2)
            cv2.rectangle(target, self.windows_right_tl[window], self.windows_right_br[window],
                          self.window_colors_r[window], 2)

    def find_lane_origins(self):
        """
        Detects the starting positions of the lane near the car at the lower half of the image.

        If no lanes could be detected or are very unlikely it uses previous remembered positions.
        If both lanes detected are very likely it remembers the street width for later calculations
        :return:
        """

        histogram = np.sum(self.cur_mask[self.cur_mask.shape[0] // 2:, :], axis=0)
        self.out_img = np.dstack((self.cur_mask, self.cur_mask, self.cur_mask)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        self.midpoint = np.int(histogram.shape[0] // 2)

        self.leftx_base = np.argmax(histogram[:self.midpoint])
        self.rightx_base = np.argmax(histogram[self.midpoint:]) + self.midpoint

        if histogram[self.leftx_base] <= self.hist_minimum_thresh and self.last_base_left is not None:
            leftx_base = self.last_base_left

        if histogram[self.rightx_base] <= self.hist_minimum_thresh and self.last_base_right is not None:
            rightx_base = self.last_base_right

        lane_width = self.rightx_base-self.leftx_base
        if lane_width>=self.valid_lane_width_min and lane_width<=self.valid_lane_width_max:
            self.last_base_left = self.leftx_base
            self.last_base_right = self.rightx_base

            # if we can right now perfectly detect both lanes recalibrate the lane size
            if histogram[self.leftx_base] >= self.hist_perf_thresh and histogram[self.rightx_base] >= self.hist_perf_thresh:
                self.lane_size = lane_width
        else:
            if self.last_base_left is not None and self.last_base_right is not None:
                self.leftx_base = self.last_base_left
                self.rightx_base = self.last_base_right

        # Identify the x and y positions of all nonzero pixels in the image
        self.nonzero = self.cur_mask.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        # Current positions to be updated for each window
        self.leftx_current = self.leftx_base
        self.rightx_current = self.rightx_base
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

    def draw_lane_lines(self):
        """
        Draws the lanes onto all output images
        """
        if len(self.left_fit_history)>0 and len(self.right_fit_history)>0:
            recent_left = self.left_fit_history[-self.interpolation:]
            recent_right = self.right_fit_history[-self.interpolation:]

            left_fit_org = np.array(recent_left).sum(axis=0)/len(recent_left)
            right_fit_org = np.array(recent_right).sum(axis=0)/len(recent_right)

            left_fit = left_fit_org
            right_fit = right_fit_org

            # Generate x and y values for plotting
            ploty = np.linspace(0, self.cur_mask.shape[0] - 1, self.cur_mask.shape[0])
            height = self.cur_mask.shape[0]
            ploty = height-ploty

            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            ploty = height-ploty

            for index, y in enumerate(ploty):
                if index>0:
                    cv2.line(self.out_img, (int(left_fitx[index-1]), int(ploty[index-1])), (int(left_fitx[index]), int(y)),
                        (255, 255, 0), 16)
                    cv2.line(self.out_img, (int(right_fitx[index - 1]), int(ploty[index - 1])), (int(right_fitx[index]), int(y)),
                         (255, 255, 0), 16)
                    cv2.line(self.out_transformed, (int(left_fitx[index - 1]), int(ploty[index - 1])), (int(left_fitx[index]), int(y)),
                             (255, 255, 0), 16)
                    cv2.line(self.out_transformed, (int(right_fitx[index - 1]), int(ploty[index - 1])),
                             (int(right_fitx[index]), int(y)),
                             (255, 255, 0), 16)

                    # points_3d_left = np.array([((left_fitx[index-1], ploty[index-1]), (left_fitx[index], y))])
                    # points_3d_left = self.transform.transform_top_perspective_coords(points_3d_left).reshape((2,2))
                    #cv2.line(self.out_perspective, (int(points_3d_left[0][0]),int(points_3d_left[0][1])) , (int(points_3d_left[1][0]),int(points_3d_left[1][1])),
                    #    (255, 0, 0), 6)
                    #points_3d_right = np.array([((right_fitx[index-1], ploty[index-1]), (right_fitx[index], y))])
                    #points_3d_right = self.transform.transform_top_perspective_coords(points_3d_right).reshape((2,2))
                    #cv2.line(self.out_perspective, (int(points_3d_right[0][0]),int(points_3d_right[0][1])) , (int(points_3d_right[1][0]),int(points_3d_right[1][1])),
                    #    (0, 0, 255), 6)

            # Create an image to draw the lines on
            warp_zero = np.zeros_like(self.cur_mask).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            pts_int = np.int_([pts])

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, pts_int , (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, self.transform.retmx , (self.original.shape[1], self.original.shape[0]))
            # Combine the result with the original image
            self.out_perspective = cv2.addWeighted(self.original, 1, newwarp, 0.3, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.curve_rad is not None:
                cv2.putText(self.out_perspective, "Estimated curve radius: {:.1f}m".format (self.curve_rad) , (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            if self.offset_to_center is not None:
                cv2.putText(self.out_perspective, "Vehicle offset to lane center: {:.1f}m".format (self.offset_to_center) , (10, 60), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def find_lanes_using_window(self, img):
        """
        Tries to detect the lanes in given image and hightlights them
        :param img: The image provided by the hero / dash cam
        :return: The images: A raw view from top, the top view as photo and the combined view
        """

        # setup images
        self.transform_original(img)
        self.select_mask_and_transform(img)

        # detect the lane positions
        self.find_lane_origins()
        self.find_windows()

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # highlight the pixels taken into account
        self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Extract left and right line pixel positions using previously estimated perfect window positions
        leftx = []
        lefty = []
        rightx = []
        righty = []

        # fit a polynomial to the detected windows
        for window in range(self.nwindows):
            left_tl = self.windows_left_tl[window]
            left_br = self.windows_left_br[window]
            right_tl = self.windows_right_tl[window]
            right_br = self.windows_right_br[window]
            leftx.append((left_tl[0]+left_br[0])/2)
            lefty.append((left_tl[1] + left_br[1]) / 2)
            rightx.append((right_tl[0] + right_br[0]) / 2)
            righty.append((right_tl[1] + right_br[1]) / 2)

        leftx = np.array(leftx)
        lefty = np.array(lefty)
        rightx = np.array(rightx)
        righty = np.array(righty)

        # Fit a second order polynomial to each
        left_fit = None
        right_fit = None

        if (leftx.shape[0] != 0 and lefty.shape[0] != 0) and (rightx.shape[0] != 0 and righty.shape[0] != 0):
            # fit top view curve lines
            height = self.cur_mask.shape[0]
            left_fit = np.polyfit(height-lefty, leftx, 2)
            self.left_fit_history.append(left_fit)
            right_fit = np.polyfit(height-righty, rightx, 2)
            self.right_fit_history.append(right_fit)

            scale_x = 3.7 / 800 # highway width in US of 3.7 meters matches about 800 pixels
            scale_y = 43 / 720
            y_eval = (self.cur_mask.shape[0]-1)*scale_y

            leftx = self.nonzerox[self.left_lane_inds]
            lefty = self.nonzeroy[self.left_lane_inds]
            rightx = self.nonzerox[self.right_lane_inds]
            righty = self.nonzeroy[self.right_lane_inds]

            # Fit polynomal to top view curves in world space
            left_fit_cr = np.polyfit(lefty*scale_y, leftx*scale_x, 2)
            right_fit_cr = np.polyfit(righty*scale_y, rightx*scale_x, 2)

            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

            self.curve_rad = (left_curverad+right_curverad)/2

            pix_off_to_center = self.cur_mask.shape[1]/2 - (self.leftx_base + self.rightx_base)/2
            self.offset_to_center = pix_off_to_center * scale_x

        self.draw_lane_lines()

        self.out_img = self.out_img.astype(np.ubyte)

        return self.out_img, self.out_transformed, self.out_perspective