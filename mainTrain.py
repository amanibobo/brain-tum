import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = []
label = []

INPUT_SIZE = 64

# Function to read and process images
def process_images(image_list, label_value, directory):
    for image_name in image_list:
        if image_name.endswith('.jpg'):
            image = cv2.imread(os.path.join(directory, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(label_value)

# Process 'no' tumor images
process_images(no_tumor_images, 0, os.path.join(image_directory, 'no'))

# Process 'yes' tumor images
process_images(yes_tumor_images, 1, os.path.join(image_directory, 'yes'))

# Print the lengths to check
print(f"Number of images: {len(dataset)}")
print(f"Number of labels: {len(label)}")

# Check if lengths match
if len(dataset) != len(label):
    raise ValueError("The number of images does not match the number of labels")

# Convert dataset into numpy array
dataset = np.array(dataset)
label = np.array(label)

# Divide into train and test
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Reshape = (n, image_width, image_height, n_channel)

# Print shapes to check
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Normalize the data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes = 2)
y_test= to_categorical(y_test, num_classes = 2)

# Model Building
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) # Output layer
model.add(Activation('softmax')) # Activation function

# Binary CrossEntropy = 1, sigmoid
# Categorical Cross Entropy = 2, softmax

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting model
model.fit(x_train, y_train, 
          batch_size=16, 
          verbose=1, epochs=10,
          validation_data=(x_test, y_test), 
          shuffle=False)

model.save('BrainTumor10EpochsCatergorical.h5')
