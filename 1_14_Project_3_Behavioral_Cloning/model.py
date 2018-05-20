
# coding: utf-8

# # Behavioral Cloning
# 
# ### Project 3 of Udacity's Self-Driving Car Engineer Nanodegree program
# 
# This is the third project of the nanodegree in which you have to train a self-driving car to adapt to a human's driving style using a convolutional neural network which has to predict the steering angle using prerecorded training data of a human driver.
# 
# ![](media/BehavioralCloning.png)

# In[2]:


from IPython.display import YouTubeVideo 
YouTubeVideo('Dx1JrIlXiDM') 


# ### Library import

# In[3]:


import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


# ## Training data configuration

# In[4]:


# learning_sets = ["Session_4_Track_1_Insane"]
#learning_sets = ["Session_4_Track_1_Insane", "Session_1_Track_1", "Session_2_Track_1_Reverse", "Session_6_Bridge", 
#                 "Session_7_Bridge", "Session_8_Bridge", "Session_9_Track_1_Curves"]
# Works, but curvey:
# learning_sets = ["Session_4_Track_1_Insane", "Session_6_Bridge", "Session_7_Bridge", "Session_8_Bridge", "Session_9_Track_1_Curves"]
# Works not, too perfect driving style:
# learning_sets = ["Session_4_Track_1_Insane", "Session_6_Bridge", "Session_7_Bridge", "Session_8_Bridge"]
# Works not, too perfect driving style:
# learning_sets = ["Session_4_Track_1_Insane", "Session_1_Track_1", "Session_2_Track_1_Reverse", "Session_6_Bridge", "Session_7_Bridge", "Session_8_Bridge"]
# Works not, too insensitive:
# learning_sets = ["Session_1_Track_1", "Session_2_Track_1_Reverse", "Session_6_Bridge", "Session_7_Bridge", "Session_8_Bridge"]
# Track 1 final - Perfect mix of driving styles:
# learning_sets = ["Session_1_Track_1", "Session_2_Track_1_Reverse", "Session_6_Bridge", "Session_7_Bridge", "Session_8_Bridge", "Session_9_Track_1_Curves"]
# Track 2:
learning_sets = ["Session_5_Track_2_Great", "Session_3_Track_2", "Session_10_Track_2_Perfect"]
#learning_sets = ["Session_1_Track_1", "Session_2_Track_1_Reverse", "Session_6_Bridge", "Session_7_Bridge", "Session_8_Bridge"]
base_data_path = "training_data/"


# ## Splitting the CSV data into file names and measurements
# 
# Here we first of all read the CSV and prepare the relative paths to all images of all training sessions.

# In[5]:


image_filenames = [] # array of tuple of images for center/left/right
measurements = [] # array of measurements

for learning_set_path in learning_sets:
    
    print("Loading {}...".format(learning_set_path))
    
    pdcsv = pd.read_csv(base_data_path+learning_set_path+"/driving_log.csv")    
    
    print(pdcsv.shape)
    
    for index, line in pdcsv.iterrows():
        source_path = line[0]
        filename_0 = source_path.split('/')[-1]
        source_path = line[1]
        filename_1 = source_path.split('/')[-1]
        source_path = line[2]
        filename_2 = source_path.split('/')[-1]
        base_path = base_data_path+learning_set_path+"/IMG/"
        image_filenames.append((base_path+filename_0, base_path+filename_1, base_path+filename_2))    
        measurement = float(line[3])
        measurements.append(measurement)
    
print("{} images and {} measurements prepared".format(len(image_filenames), len(measurements)))

center_example = len(image_filenames)//2


# In[6]:


def get_full_training_image_path(index, direction):
    """Returns the full path to an image at given index and using a specific direction
    
    index: The image index
    direction: The camera index. 0 = straight view camera, 1 = left side, 2 = right side
    """
    return image_filenames[index][direction]


# In[7]:


print(get_full_training_image_path(center_example,0))


# ## Presenting example images
# 
# Below you can see an image of all 3 cameras, the straight camera, the left camera and the right camera.

# In[8]:


columns = 3
rows = 1
total_samples = columns*rows

fig = plt.figure(figsize=(columns*10,rows*10))

for i in range(total_samples):
    ax = fig.add_subplot(rows, columns, i + 1, xticks=[], yticks=[])
    filename = get_full_training_image_path(center_example,i)
    cur_org = cv2.imread(filename)    
    b,g,r = cv2.split(cur_org)       # get b,g,r
    cur_org = cv2.merge([r,g,b])     # switch it to rgb
    
    ax.imshow(cur_org)


# ## Image loading and augmentation
# 
# Here we load all images to memory. To increase the amount of training data and be better prepared for steering corrections we use the images of all 3 "mounted" cameras with a slight adjustment of the desired angle to be predicted.

# In[9]:


augmented_images = []
augmented_measurements = []

for index, ifn in enumerate(image_filenames):
    for side in range(1):
        image = cv2.imread(get_full_training_image_path(index,side))
        b,g,r = cv2.split(image)       # get b,g,r
        image = cv2.merge([r,g,b])     # switch it to rgb
        measurement = measurements[index]
        if side==1:
            measurement += 0.1
        if side==2:
            measurement -= 0.1
            
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(-measurement)
    
print(len(augmented_images))
print(augmented_images[0].shape)


# In[10]:


columns = 3
rows = 1
total_samples = columns*rows

fig = plt.figure(figsize=(columns*10,rows*10))

for i in range(total_samples):
    ax = fig.add_subplot(rows, columns, i + 1, xticks=[], yticks=[])
    filename = get_full_training_image_path(center_example,i)
    cur_org = augmented_images[100]
    
    ax.imshow(cur_org)


# ## Image cropping test
# 
# To let the network focus on the really important parts of the image we crop away the upper and lower part of the image as the clouds likely won't help us to better predict the perfect steering angle.

# In[11]:


columns = 3
rows = 1
total_samples = columns*rows

crop_percent_top = 0.2
crop_percent_bottom = 0.2

fig = plt.figure(figsize=(columns*10,rows*10))

for i in range(total_samples):
    ax = fig.add_subplot(rows, columns, i + 1, xticks=[], yticks=[])
    filename = get_full_training_image_path(center_example,i)
    cur_org = augmented_images[100]
    
    width = cur_org.shape[1]
    height = cur_org.shape[0]
    top_off = int(crop_percent_top*height)
    rem_height = int(height-(crop_percent_bottom+crop_percent_top)*height)
    cur_org = cur_org[top_off:top_off+rem_height, :, :]    
    
    ax.imshow(cur_org)
    
print(augmented_images[0].shape)


# ## Training and exporting the model
# 
# When everything is prepared we feed all of this data into a convolutional neural network with 5 convolutional and 4 fully connected layers.
# 
# Instead of using the neural network to try to categorize and image or detect something within it we use it to predict the steering angle. So for example if it detects lane lines pretty close the the left side of the vehicle it shall steer to the right and if it detects ones close to the right side it shall steer to the left. If it's perfectly in the center of the road it shall not steer at all.

# In[12]:


# Convert the training data to numpy tensors

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# Setup a sequential, convolutional network which predicts the optimum steering angle in different situations
model = Sequential()
# The original input image is cropped and normalized. Storing this in the model makes it possible to automatically
# crop the image from the simulator as well
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60,12),(0,0))))
# 5 convolutional layers, nVidia style, first three layers use a 5x5, the other two ones a 3x3 kernel
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
# Flatten the data and add another 3 fully connected layers
model.add(Flatten())
# To prevent overfitting enable dropout of 25%
model.add(Dropout(0.25))
model.add(Dense(100))
# To prevent overfitting enable dropout of 25%
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam())

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=8)

model.save('model.h5')


# In[22]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def plot_keras_model(model, show_shapes=True, show_layer_names=True):
    return SVG(model_to_dot(model, show_shapes=show_shapes, show_layer_names=show_layer_names).create(prog='dot',format='svg'))


file = open('model.svg', 'w')
file.write(svg_data.data)
file.close()

svg_data = plot_keras_model(model, show_shapes=True, show_layer_names=False)
plot_keras_model(model, show_shapes=True, show_layer_names=False)

