# **Finding Lane Lines on the Road** 

This is the first project of Udacity's Self-Driving Car Nanodegree in which the task is to detect the lanes of a street using computer vision.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image_res]: test_image_out/combined.png "Result"
[image_gray]: test_image_out/gray.png "Gray"
[image_gaussian]: test_image_out/gaussian.png "Filtered"
[image_canny]: test_image_out/canny_edges.png "Canny applied"
[image_masked]: test_image_out/masking.png "Masked"
[image_raw_hough]: test_image_out/hough_raw.png "Raw hough"
[image_clean_hough]: test_image_out/hough_filtered.png "Cleaned hough"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consits of 6 steps:

**Step 1 - Grayscale**

The input image is converted from RGB to grayscale to prepare it for the sobel (edge highlighting) filter of step 3.
![image_gray]

**Step 2 - Filtering**

To remove camera noise and to smooth the result of step 3 we apply a gaussian blur filter.
![image_gaussian]

**Step 3 - Edge detection using canny edge detection**

Now we are ready to apply our canny filter which first of all applies a sobel filter to our blured grayscale image and then highlights the most dominant edges as thin lines.

For more details see: [https://en.wikipedia.org/wiki/Canny_edge_detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
![image_canny]

**Step 4 - Clipping**

To remove the horizon, offroad elements and neighbour lanes we now apply a trapezoid mask to our image and remove these elements.
![image_masked]

**Step 5 - Line creation**

In this step we use the OpenCVs hough algorithm which converts all thin highlighted lines of step 3 into real lines and connects them with help of a polar coordinate hough space.
All lines with a given minimum length and skipping smaller gaps are combined to larger lines. Also I removed all nearly completely horizontal lines in this step as well.

For more details see [https://en.wikipedia.org/wiki/Hough_transform](https://en.wikipedia.org/wiki/Hough_transform)

![image_raw_hough]

**Step 6 - Cleanup (Modification of the draw_lines function)**

In this step I first of all sort all lines of step 2 into two groups using theirs direction into either the left lane marker group or the right lane marker group.
After the sorting the average direction and the extended position right next to the vehicle is calculatated and combined into one single left side and one single right side lane.
![image_clean_hough]

**Result**

Below you can see the cleaned up lines of step 6 as an overlay on the original image.

![image_res]

**Action**

Below you can see the whole technique in action

<video src="test_videos_output/solidYellowLeft.mp4" width=600 controls>

### 2. Identify potential shortcomings with your current pipeline

The algorithm works pretty fine one relative straight roads, but the more curvey they become the more less suboptimal the current trapezoid mask and the cleanup method of step 6 which averages the directions is.

Also the algorithm at the moment counts on that there is always a left and always a right line which is not guaranteed.

Another shortcoming is that it directly in the first step discards all color information and is so not able to priorize temporary (yellow or orange) lane markers above white ones anymore.

### 3. Suggest possible improvements to your pipeline

At the moment the algorithm counts on relative straight lines. One way to also support curvey roads would be estimating the next line marker by the output direction of the previous ones as the direction only changes step by step from marker to marker.

A solution for yellow and orange markers would likely be to apply the algorithm multiple times and to then predict how likely the yellow and orange markers are valid (overall lengths, distance to lane etc.) and let them replace the white ones. (for example in construction zones).
