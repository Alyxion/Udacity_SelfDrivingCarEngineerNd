#####################################################################################################################
#                                                                                                                   #
# This file is part of the 5th project of Udacity's Self-Driving Car Engineer Nd - Vehicle Detection and Tracking   #
#                                                                                                                   #
# Copyright (c) 2018 by Michael Ikemann                                                                             #
#                                                                                                                   #
#####################################################################################################################

import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import time
import cv2
import Hog_Me
import pickle

class Object_Classifier:
    """
    The Object_Classifier trains a support vector machine or neural network to discriminate between photos of
    two given classes, using positive and negative samples.

    Properties:

    hogger = The Hoggit object which converts images into HOGs
    predictor = The trained predictor which can predict if a photo contains the object previously trained
    X_scaler = Normalizes the hog data before feeding it into the svm or nn
    box_size = The size of the images to be predicted
    may_flip_horizontal = Defines if training images may be flipped horizontally
    """

    def __init__(self, hogger, use_svm = False):
        """
        Constructor
        :param hogger: The hog creator. See Hoggit
        :param use_svm: Defines if a svm shall be used. Otherwise a neural network will be used. (default = nn)
        """
        self.hogger = hogger
        self.predictor = None
        self.X_scaler = None
        self.use_svm = use_svm
        self.box_size = 0
        self.may_flip_horizontal = False

    def train(self, positive_images, negative_images):
        """
        Trains the svm or neural network
        :param positive_images: A list of positive image file names (such as vehicles)
        :param negative_images: A list of negative image file names (such as the street, lane lines etc.)
        """
        self.positive_features = self.get_features(positive_images)
        self.negative_features = self.get_features(negative_images)

        # Create an array stack of feature vectors
        X = np.vstack((self.positive_features, self.negative_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(self.positive_features)), np.zeros(len(self.negative_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        print('Feature vector length:', len(X_train[0]))
        if self.use_svm:
            # Use a linear SVC
            print("Using support vector machine")
            self.predictor = LinearSVC()
        else:
            # Use neural network
            print("Using neural network")
            self.predictor = MLPClassifier(hidden_layer_sizes=(100, ), activation="relu", solver="adam")
        # Check the training time for the SVC
        t = time.time()
        self.predictor.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train predictor...')
        # Check the score of the SVC
        print('Test Accuracy of predictor = ', round(self.predictor.score(X_test, y_test), 4))

    def get_features(self, file_list):
        """
        Returns a stack of features for a whole list of files provided
        :param file_list: The file list
        :return: An array of feature arrays
        """
        all_features = []

        for file_name in file_list:
            img = cv2.imread(file_name)
            org_image = img

            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            features, img = self.hogger.hog_image(rgb_image, visualize = False)
            all_features.append(features)

            if self.may_flip_horizontal:    # Add horizontal mirrored version as well if allowed
                features, img = self.hogger.hog_image(cv2.flip(rgb_image, 0), visualize=False)
                all_features.append(features)

        self.box_size = org_image.shape[0]

        return all_features

    def classify(self, image):
        """
        Classifies an image by converting it into HOGs as well and then using the trained classifier.
        :param image: The image
        :return: The prediction (1.0 = perfect match)
        """
        features, img = self.hogger.hog_image(image, visualize=False)

        all_features= []
        all_features.append(features)

        X = np.vstack(all_features).astype(np.float64)
        X = self.X_scaler.transform(X)

        return self.predictor.predict(X)

    def classify_features(self, features):
        """
        Classifies an image by using an already prepares feature list (for example the sub region of a larget image)
        :param features: The hog features
        :return: The prediction (1.0 = perfect match)
        """
        all_features= []
        all_features.append(features)

        X = np.vstack(all_features).astype(np.float64)
        X = self.X_scaler.transform(X)

        return self.predictor.predict(X)

    def save_to_disk(self, file_name):
        """
        Saves the trained classifier to disk so it can be loaded later in another script
        :param file_name: The target file name
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.predictor, f)
            pickle.dump(self.X_scaler, f)
            pickle.dump(self.box_size, f)

    def load_from_disk(self, file_name):
        """
        Loads a previously trained classifier from disk
        :param file_name: The source file name
        """
        with open(file_name, 'rb') as f:
            self.predictor = pickle.load(f)
            self.X_scaler = pickle.load(f)
            self.box_size = pickle.load(f)