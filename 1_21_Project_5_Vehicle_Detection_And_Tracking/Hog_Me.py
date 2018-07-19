import numpy as np
from skimage.feature import hog
import cv2
import math

class Hoggit:

    def __init__(self):
        self.orientations = 12
        self.pix_per_cell = 16
        self.cells_per_block = 2
        self.hog_features = None
        self.vis_image = None
        self.last_image = None
        self.hog_channel = "ALL"

    def hog_image(self, img, visualize = True, feature_vector = True):
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.cells_x = self.width // self.pix_per_cell
        self.cells_y = self.height // self.pix_per_cell

        self.x_blocks = self.cells_x - self.cells_per_block + 1
        self.y_blocks = self.cells_y - self.cells_per_block + 1
        self.total_blocks = self.x_blocks * self.y_blocks
        self.total_features = self.total_blocks * self.cells_per_block * self.cells_per_block * self.orientations

        img= cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        if self.hog_channel == 'ALL':
            features = []
            for channel in range(img.shape[2]):
                features.extend(hog(img[:,:,channel], orientations=self.orientations,
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cells_per_block, self.cells_per_block),
                           block_norm='L2-Hys',
                           transform_sqrt=False,
                           visualise=visualize, feature_vector=feature_vector))
            hog_img = None
        else:
            if visualize:
                features, hog_img = hog(img[:,:,self.hog_channel], orientations=self.orientations,
                                          pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                          cells_per_block=(self.cells_per_block, self.cells_per_block),
                                          block_norm='L2-Hys',
                                          transform_sqrt=False,
                                          visualise=visualize, feature_vector=feature_vector)
            else:
                features = hog(img[:,:,self.hog_channel], orientations=self.orientations,
                               pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                               cells_per_block=(self.cells_per_block, self.cells_per_block),
                               block_norm='L2-Hys',
                               transform_sqrt=False,
                               visualise=visualize, feature_vector=feature_vector)
                hog_img = None

        self.vis_image = np.copy(hog_img) if hog_img is not None else None
        self.hog_features = np.copy(features)

        return features, hog_img

    def get_sub_region(self, x, y, width, height):
        # Note: Work in progress, not used in submission

        if x<0 or y<0 or x+width>self.width or y+height>self.height:
            return None

        cells_x = width // self.pix_per_cell
        cells_y = height // self.pix_per_cell
        x_blocks = cells_x - self.cells_per_block + 1
        y_blocks = cells_y - self.cells_per_block + 1
        cells_per_block_tot = self.cells_per_block*self.cells_per_block
        total_blocks = x_blocks * y_blocks
        x_col = x//self.pix_per_cell
        y_row = y//self.pix_per_cell
        per_row = self.x_blocks * self.cells_per_block * self.cells_per_block * self.orientations
        y_offset = y_row * per_row
        x_offset = x_col * self.cells_per_block * self.orientations

        result = []

        s_off = y_offset+x_offset
        for y_ind in range(y_blocks):
            c_off = s_off + y_ind*per_row
            result.append(self.hog_features[c_off:c_off+x_blocks*cells_per_block_tot*self.orientations])

        return np.hstack(result)

    def visualize_features(self, features, width, height):
        # Note: Work in progress, not used in submission
        img = np.zeros((height, width, 3), np.uint8)

        print(features.shape)

        cells_x = width // self.pix_per_cell
        cells_y = height // self.pix_per_cell
        x_blocks = cells_x - self.cells_per_block + 1
        y_blocks = cells_y - self.cells_per_block + 1
        cells_per_block_tot = self.cells_per_block*self.cells_per_block
        entries_per_block = self.cells_per_block*self.cells_per_block*self.orientations
        total_blocks = x_blocks * y_blocks

        per_row = x_blocks * entries_per_block

        gradients = np.zeros((height,width,self.orientations))

        print(gradients.shape)

        for row in range(y_blocks):
            for col in range(x_blocks):
                off = row * per_row + col * entries_per_block
                for orientation in range(self.orientations):
                    gradients[row][col][orientation] = features[off+orientation]

        print("Painting")

        return self.render_hog(img, gradients)

    def render_hog(self, image, cell_gradient):
        # Note: Work in progress, not used in submission
        cell_size = self.pix_per_cell/ 2
        max_mag = np.array(cell_gradient).max()
        print(max_mag)

        per_angle =  math.pi/self.orientations

        for y in range(cell_gradient.shape[0]):
            for x in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[y][x]
                cell_grad /= max_mag
                angle = 0
                for magnitude in cell_grad:
                    cv = math.cos(angle)
                    sv = math.sin(angle)
                    x1 = int(x * self.pix_per_cell + magnitude * cell_size * cv)
                    y1 = int(y * self.pix_per_cell + magnitude * cell_size * sv)
                    x2 = int(x * self.pix_per_cell - magnitude * cell_size * cv)
                    y2 = int(y * self.pix_per_cell - magnitude * cell_size * sv)
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * magnitude))
                    angle += per_angle

        return image