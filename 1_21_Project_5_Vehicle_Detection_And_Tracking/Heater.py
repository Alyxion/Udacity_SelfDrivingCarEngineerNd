import cv2
import numpy as np
from scipy.ndimage.measurements import label

class Heater:

    def __init__(self):
        self.history = []
        self.max_remembering = 3
        self.base_scale = 5
        self.labels = []
        self.min_size = 60
        self.threshold = self.max_remembering*self.base_scale*3

    def add_to_history(self, rectangles):
        self.history.append(rectangles)

    def add_heat(self, heatmap):
        # Iterate through list of bboxes

        look_back = self.max_remembering
        hist_start = len(self.history)-1
        if hist_start-look_back+1<0:
            look_back = hist_start+1

        for hist_index in range(hist_start-look_back+1,hist_start+1):
            for box in self.history[hist_index]:
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += int(self.base_scale*self.max_remembering/look_back)

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def draw_labeled_bboxes(self, img):
        for bbox in self.labels:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 2)

        return img

    def get_heat(self, image):
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = self.add_heat(heat)
        color_heat = np.zeros_like(image[:, :, :]).astype(np.float)

        heat[heat <= self.threshold] = 0
        color_heat[heat!=0] = (170,0,0)
        color_heat = color_heat.astype(np.uint8)

        new_labels = label(heat)
        self.labels = []
        for car_number in range(1, new_labels[1] + 1):
            nonzero = (new_labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            width = bbox[1][0] - bbox[0][0]
            height= bbox[1][1] - bbox[0][1]
            if (width > self.min_size) and (height > self.min_size):
                self.labels.append(bbox)

        return heat, color_heat
