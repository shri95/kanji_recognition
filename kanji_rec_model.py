# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:58:38 2016

@author: shri

This file contains the model information. This is to be shared between train and test process
"""
###################################################
import tensorflow as tf
SEED = 66478  # Set to None for random seed.

from kanji_rec_common_params import *

# Variable that hold model parameters
conv1_weights = tf.Variable(
      tf.truncated_normal([7, 7, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
conv1_biases = tf.Variable(tf.zeros([32]))
conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 128],
                          stddev=0.1,
                          seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[128]))
fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [IMAGE_SIZE // 16 * IMAGE_SIZE // 16 * 128, 512],
          stddev=0.1,
          seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
fc2_weights = tf.Variable(
      tf.truncated_normal([512, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

# CNN based model definition
def model(data, train=False):
    """The Model definition."""
    # Convolutional layer 1 : size 7x7x32
#    print("input",data.get_shape().as_list())
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
                        
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # Max pooling. Pooling window is 4x4 and stride is also 4x4
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1],
                          padding='SAME')
    # Convolutional layer 2 : kernel size 5x5x128
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

    pool = tf.nn.max_pool(relu,
                          ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1],
                          padding='SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        
    # Fully connected layer. 
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    
    returnVal = tf.matmul(hidden, fc2_weights) + fc2_biases
    
    return returnVal