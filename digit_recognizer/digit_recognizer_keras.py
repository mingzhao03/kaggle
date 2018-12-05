#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:07:18 2018

Uses focal loss [1]. Taylor softmax [2] is used for numerical stability
[1] arXiv:1708.02002v2  [cs.CV]  7 Feb 2018
[2] arXiv:1511.05042v3  [cs.NE]  28 Feb 2016

@author: mzhao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

#%%
train_path = './train.csv'
test_path = './test.csv'
image_size = [28, 28]
num_labels = 10
num_validation_samples = 1000
initial_learning_rate = 0.001
learning_rate_decay = 0.99
gamma = 3
training_epochs = 400
batch_size = 1000

num_rotations = 7

train_data = pd.read_csv(train_path).values.astype(np.float32)
train_labels, train_images = np.split(train_data,[1],axis=1)
train_images = np.reshape(train_images, (-1, 28, 28, 1)) / 255.

test_images = pd.read_csv(test_path).values.astype(np.float32)
test_images = np.reshape(test_images, (-1, 28, 28, 1)) / 255.

np.random.seed(0)
from tensorflow import random
random.set_random_seed(0)

#%% Define custom objective function with Taylor softmax and focal loss
def focal_objective(y_true, y_pred):
    taylor_ex = 1. + y_pred + 0.5 * K.pow(y_pred, 2)
    taylor_sum = K.reshape(K.sum(taylor_ex, axis=1, keepdims=False), shape=[-1, 1])
    taylor_softmax = taylor_ex / taylor_sum
    cost = -K.mean(K.sum(K.cast(y_true, 'float32') * K.log(taylor_softmax)
            * K.pow(1. - taylor_softmax, gamma), axis=1, keepdims=False), axis=0, keepdims=False)
    return cost

#%% Set up train set and validation set
num_train_samples = train_data.shape[0] - num_validation_samples
arr = np.arange(train_data.shape[0])
np.random.shuffle(arr)
train_set_images = train_images[arr[0:num_train_samples],:]
train_set_labels = to_categorical(train_labels[arr[0:num_train_samples],:], num_classes=num_labels)
validation_set_images = train_images[arr[num_train_samples:],:]
validation_set_labels = to_categorical(train_labels[arr[num_train_samples:],:], num_classes=num_labels)

#%% define Keras CNN model
model = Sequential()

model.add(Lambda(lambda x: x**0.25, input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(7,7),strides=(1,1), padding='Same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5),strides=(1,1), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5,5),strides=(1,1), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1), padding='Same', activation='relu',))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation=None))

optimizer = Adam(lr=initial_learning_rate, decay=0)

model.compile(optimizer=optimizer, loss=focal_objective, metrics=['accuracy'])

#%% image augmentation
gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#datagen.fit(train_set_images)

batches = gen.flow(train_set_images, train_set_labels, batch_size=batch_size)
validation_batches = gen.flow(validation_set_images, validation_set_labels, batch_size=batch_size)

#num_steps = num_train_samples // batch_size

hist = model.fit_generator(generator=batches, steps_per_epoch=batches.n // batch_size, epochs=training_epochs, 
                           validation_data=validation_batches, validation_steps=validation_batches.n // batch_size)

#%%
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(hist.history['loss'], color='b', label="Training loss")
ax[0].plot(hist.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(hist.history['acc'], color='b', label="Training accuracy")
ax[1].plot(hist.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

#%% Apply the model
results = model.predict(test_images)
results = np.argmax(results, axis=1)

imageId = np.arange(1, results.shape[0]+1)
df = pd.DataFrame({'ImageId':imageId,'Label':results})
df.to_csv('./cnn_keras.csv',index=False)
