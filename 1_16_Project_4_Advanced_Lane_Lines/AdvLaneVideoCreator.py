#####################################################################################################################
#                                                                                                                   #
# This file is part of the 4th project of Udacity's Self-Driving Car Engineer project Advanced Lane Finding         #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import AdvLaneCamera
import AdvLaneHelper
import AdvLanePerspectiveTransform
import AdvLaneThresher
import AdvLaneFinder

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# ---------------- Setup ----------------

camera = AdvLaneCamera.AdvCamera()
camera.load_from_pickle()
lane_helper = AdvLaneHelper.AdvLaneHelper(camera)
thresher = AdvLaneThresher.AdvLaneThresher()
example_images = lane_helper.get_example_images()

example_image = lane_helper.load_and_undistort(example_images[6])
perspective_transform = AdvLanePerspectiveTransform.LanePerspectiveTransform(example_image)

lane_finder = AdvLaneFinder.LaneFinder(camera, perspective_transform, thresher)

# ---------------- Video creation ----------------

def create_top_video():
    """
    Creates a video which contains the estimated lane line markers using the binary top view
    :return:
    """

    def process_image(image):
        warped, warped_cam, perspec = lane_finder.find_lanes_using_window(image)
        return warped

    find_lanes_raw = 'test_videos_output/find_lanes_raw.mp4'

    project_video = "project_video.mp4"

    white_output = find_lanes_raw
    clip1 = VideoFileClip(project_video)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)


def create_top_video_photo():
    """
    Creates a video which contains the estimated lane line markers using a top view and the top view image
    :return:
    """

    def process_image(image):
        warped, warped_cam, perspec = lane_finder.find_lanes_using_window(image)
        return warped_cam

    find_lanes_raw = 'test_videos_output/find_lanes_raw_photo.mp4'

    project_video = "project_video.mp4"

    white_output = find_lanes_raw
    clip1 = VideoFileClip(project_video)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

def create_perspective_video():
    """
    Creates the final submission video
    :return:
    """

    def process_image(image):
        warped, warped_cam, perspec = lane_finder.find_lanes_using_window(image)
        return perspec

    combined = 'test_videos_output/combined.mp4'

    project_video = "project_video.mp4"

    white_output = combined
    clip1 = VideoFileClip(project_video)
    # clip1 = clip1.subclip(0.0, 1.0)

    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

# create_top_video_photo()
# create_perspective_video()