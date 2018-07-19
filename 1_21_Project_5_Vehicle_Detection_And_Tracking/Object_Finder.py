import Object_Classifier
import Hog_Me
import cv2
import numpy as np

class Object_Finder:

    def __init__(self, classifier, hogger):
        self.classifier = classifier
        self.box_size = classifier.box_size
        self.boundings = []
        self.single_hog = True
        self.hogger= hogger
        self.resized_image = None

    def get_resized_sub_sample(self, off_x, off_y):

        sub_sample = self.original[off_y:off_y + self.eff_box_size, off_x:off_x + self.eff_box_size, :]
        new_size = (self.box_size, self.box_size)
        return cv2.resize(sub_sample, new_size)

    def select(self, image):
        self.original = image
        self.boundings = []

    def set_scaling(self, scaling):
        self.scaling = scaling
        self.eff_box_size = int(self.box_size*self.scaling+0.5)

    def find_instances_in_features(self, features, region):
        pix_per_cell = 16
        cell_per_block = 2
        orient = 12

        offset_per_channel = features.shape[0]//3

        block_size = cell_per_block*pix_per_cell
        x_range = self.box_size//pix_per_cell - cell_per_block + 1
        y_range = self.box_size//pix_per_cell - cell_per_block + 1

        x_steps = features.shape[1]-x_range+1
        y_steps = offset_per_channel-x_range+1
        step_factor = 1

        for y_off in range(y_steps):
            for x_off in range(x_steps):
                xpos = x_off*step_factor
                ypos = y_off*step_factor

                hog_feat1 = features[ypos:ypos+y_range, xpos:xpos+x_range].ravel()
                hog_feat2 = features[offset_per_channel+ypos:offset_per_channel+ypos+y_range, xpos:xpos+x_range].ravel()
                hog_feat3 = features[2*offset_per_channel+ypos:2*offset_per_channel+ypos+y_range, xpos:xpos+x_range].ravel()

                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                if self.classifier.classify_features(hog_features)==1.0:
                    off_x = int(xpos * pix_per_cell)
                    off_y = int(ypos * pix_per_cell)

                    trans_off_x = int(xpos * pix_per_cell * self.scaling) + region[0]
                    trans_off_y = int(ypos * pix_per_cell * self.scaling) + region[1]

                    cv2.rectangle(self.resized_image, (off_x, off_y), (off_x + self.box_size, off_y + self.box_size),
                                  color=(255, 255, 255), thickness=3)
                    cv2.rectangle(self.image, (trans_off_x, trans_off_y), (trans_off_x + self.eff_box_size, trans_off_y + self.eff_box_size),
                                  color=(255, 255, 255), thickness=3)
                    self.boundings.append(((trans_off_x, trans_off_y), (trans_off_x + self.eff_box_size, trans_off_y + self.eff_box_size)))

    def find_instances(self, image, region, overlap):

        self.image = np.copy(image)

        self.eff_step_size = int((1.0-overlap)*self.eff_box_size)

        y_steps = (region[3]-region[1])//self.eff_step_size
        x_steps = (region[2]-region[0])//self.eff_step_size

        if region[0]+(x_steps-1)*self.eff_step_size+self.eff_box_size>region[2]:
            x_steps -= 1
        if region[1]+(y_steps-1)*self.eff_step_size+self.eff_box_size>region[3]:
            y_steps -= 1

        if self.single_hog:

            self.resized_image = image[region[1]:region[3],region[0]:region[2],:]
            self.resized_image = cv2.resize(self.resized_image, (int(self.resized_image.shape[1]/self.scaling), int(self.resized_image.shape[0]/self.scaling)))
            features, img = self.hogger.hog_image(self.resized_image, visualize=False, feature_vector=False)
            features = np.array(features)

            self.find_instances_in_features(features, region)

            return self.image, self.resized_image

        else:
            for row in range(y_steps):
                off_y = region[1] + row * self.eff_step_size
                for col in range(x_steps):
                    off_x = region[0]+col * self.eff_step_size

                    sub_sample = self.get_resized_sub_sample(off_x, off_y)

                    pred = self.classifier.classify(sub_sample)

                    if(pred==1.0):
                        cv2.rectangle(self.image, (off_x, off_y), (off_x+self.eff_box_size, off_y+self.eff_box_size), color=(255,255,255), thickness=3)
                        self.boundings.append(((off_x, off_y), (off_x+self.eff_box_size, off_y+self.eff_box_size)))

            return self.image, None