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
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#%% Initialize parameters
train_path = './train.csv' 
test_path = './test.csv'
image_size = [28, 28]
num_labels = 10
num_validation_samples = 1000
initial_learning_rate = 0.0005
learning_rate_decay = 0.99
gamma = 3
training_epochs = 200
batch_size = 1000

num_rotations = 7

train_data = pd.read_csv(train_path).values.astype(np.float32)
train_labels, train_images = np.split(train_data,[1],axis=1)
train_images = train_images / 255.

test_images = pd.read_csv(test_path).values.astype(np.float32)
test_images = test_images / 255.

np.random.seed(0)
tf.random.set_random_seed(0)

#%% Set up train set and validation set
num_train_samples = train_data.shape[0] - num_validation_samples
arr = np.arange(train_data.shape[0])
np.random.shuffle(arr)
train_set_images = train_images[arr[0:num_train_samples],:]
train_set_labels = train_labels[arr[0:num_train_samples],:]
validation_set_images = train_images[arr[num_train_samples:],:]
validation_set_labels = train_labels[arr[num_train_samples:],:]

#%% define tf model
x = tf.placeholder(tf.float32, shape=[None, image_size[0]*image_size[1]])
y = tf.placeholder(tf.int32, shape=[None, num_labels])
isTraining = tf.placeholder_with_default(True, shape=())
rotate_angle = tf.placeholder(tf.float32, shape=[None])

x_reshaped = tf.reshape(x, shape=[-1, image_size[0], image_size[1], 1])
x_root = tf.pow(x_reshaped, 0.25)

conv1 = tf.layers.conv2d(inputs=x_root, filters=32, kernel_size=[7,7], padding='same', activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1, filters = 32, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
dropout1 = tf.layers.dropout(inputs=pool1, rate=0.25, training=isTraining)

conv3 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[5,5], padding='same', activation=tf.nn.relu)
conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)
dropout2 = tf.layers.dropout(inputs=pool2, rate=0.25, training=isTraining)

pool2_flat = tf.reshape(dropout2, [-1, 7*7*64])

dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
dropout3 = tf.layers.dropout(inputs=dense, rate=0.5, training=isTraining)
logit = tf.layers.dense(inputs=dropout3, units=num_labels)

taylor_ex = tf.add(1., tf.add(logit, 0.5 * tf.pow(logit, 2)))
taylor_sum = tf.reshape(tf.reduce_sum(taylor_ex, axis=1), shape=[-1, 1])
taylor_softmax = tf.div(taylor_ex , taylor_sum)
cost = -tf.reduce_mean(tf.reduce_sum(tf.cast(y, tf.float32) * tf.log(taylor_softmax) * tf.pow(1. - taylor_softmax, gamma), 1))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

learning_rate = tf.placeholder(tf.float32, shape=[])
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

pred = tf.argmax(logit, 1)
correct_pred = tf.equal(pred, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Data augmentation operations
tf_rotate_images = tf.reshape(tf.contrib.image.rotate(x_reshaped, rotate_angle, interpolation='BILINEAR'), 
                              [-1, image_size[0]*image_size[1]])

#%% Get session and start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    onehot_train_labels = tf.squeeze(tf.one_hot(train_set_labels, num_labels, on_value=1., off_value=0., axis=-1))
    onehot_train_vals = sess.run(onehot_train_labels)
    onehot_validation_labels = tf.squeeze(tf.one_hot(validation_set_labels, num_labels, on_value=1., off_value=0., axis=-1))
    onehot_validation_vals = sess.run(onehot_validation_labels)
    
    num_batches = num_train_samples // batch_size
    current_learning_rate = initial_learning_rate
    
    for epoch in range(training_epochs):
        # randomize data
        arr = np.arange(num_train_samples)
        np.random.shuffle(arr)
        for batch in range(num_batches):
            batch_images = train_set_images[arr[batch*batch_size:(batch+1)*batch_size],:]
            batch_labels = onehot_train_vals[arr[batch*batch_size:(batch+1)*batch_size],:]

            # Image augmentation with rotations.
            for a in range(num_rotations):
                angle = np.random.uniform(-0.6, 0.6, batch_size)
                rotated_images = sess.run(tf_rotate_images, feed_dict={x:batch_images, rotate_angle: angle})
                _, acc_val = sess.run([train_op, accuracy], feed_dict={x:rotated_images, y:batch_labels, 
                                      isTraining:True, learning_rate:current_learning_rate})

            _, acc_val = sess.run([train_op, accuracy], feed_dict={x:batch_images, y:batch_labels, 
                                  isTraining:True, learning_rate:current_learning_rate})                
    
        print('Epoch', epoch, 'train set accuracy', acc_val)
        acc_val, cost_val = sess.run([accuracy, cost], feed_dict={x:validation_set_images, y:onehot_validation_vals, isTraining:False})
        print('Epoch', epoch, 'test set accuracy', acc_val,'cost', cost_val)
        current_learning_rate *= learning_rate_decay

    pred_val, correct_vals, img_reshaped, acc_val = sess.run([pred, correct_pred, x_reshaped, accuracy], feed_dict={x:validation_set_images, y:onehot_validation_vals, isTraining:False})

    # plot wrong predictions
    plt.figure()
    num_rows = 3
    num_cols = 6
    plot_index = 1
    for i in range(len(correct_vals)):
        if (not correct_vals[i]):
            plt.subplot(num_rows, num_cols, plot_index)
            plt.title(str(validation_set_labels[i]) + str(pred_val[i]))
            plt.imshow(np.squeeze(img_reshaped[i,:,:,:]))
            plt.axis('off')
            plot_index += 1

    # make prediction
    predictions = []
    num_batches = test_images.shape[0] // batch_size
    for batch in range(num_batches):
        batch_test_images = test_images[batch*batch_size:(batch+1)*batch_size,:]
        pred_test = sess.run(pred, feed_dict={x:batch_test_images, isTraining:False})
        predictions.append(pred_test)
    
    result = np.concatenate(predictions)
imageId = np.arange(1, result.shape[0]+1)
df = pd.DataFrame({'ImageId':imageId,'Label':result})
df.to_csv('./cnn_submission2.csv',index=False)
