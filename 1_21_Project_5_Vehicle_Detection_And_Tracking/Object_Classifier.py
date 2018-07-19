import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import time
import cv2
import Hog_Me

class Object_Classifier:

    def __init__(self, hogger):
        self.hogger = hogger
        self.predictor = None
        self.box_size = 0

    def train(self, positive_images, negative_images):

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
        # Use a linear SVC
        #self.predictor = LinearSVC()
        # Use neural network
        self.predictor = MLPClassifier()
        # Check the training time for the SVC
        t = time.time()
        self.predictor.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train neural network...')
        # Check the score of the SVC
        print('Test Accuracy of predictor = ', round(self.predictor.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()

    def get_features(self, file_list):

        all_features = []

        for file_name in file_list:
            img = cv2.imread(file_name)
            org_image = img

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            features, img = self.hogger.hog_image(img, visualize = False)

            all_features.append(features)

        self.box_size = org_image.shape[0]

        return all_features

    def classify(self, image):
        features, img = self.hogger.hog_image(image, visualize=False)

        all_features= []
        all_features.append(features)

        X = np.vstack(all_features).astype(np.float64)
        X = self.X_scaler.transform(X)

        return self.predictor.predict(X)

    def classify_features(self, features):
        all_features= []
        all_features.append(features)

        X = np.vstack(all_features).astype(np.float64)
        X = self.X_scaler.transform(X)

        return self.predictor.predict(X)