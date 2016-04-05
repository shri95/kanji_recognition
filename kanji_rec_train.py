# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 17:35:22 2016

@author: shri
"""

"""Based on MNIST example.
"""

import sys
import time

import numpy
from six.moves import xrange
import tensorflow as tf
import numpy as np
from kanji_rec_common_params import *
from kanji_rec_model import *

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'


VALIDATION_SIZE = 5000  # Size of the validation set.

BATCH_SIZE = 128*6
EVAL_BATCH_SIZE = 128*6
EVAL_FREQUENCY = 10  # Number of steps between evaluations.
TRAIN_STEPS = 100000 


def extract_data(filename, num_images):
  """ The data samples are picked from the numpy data file. The data is organized in 
  [NUM_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, 1] by processing it offline. """
  print('Extracting', filename)
  nparr = np.load(filename)
  nparr = np.float32(nparr)
  nparr = (nparr - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  nparr = nparr.reshape([num_images,IMAGE_SIZE,IMAGE_SIZE,1])

  return nparr
  

def extract_labels(filename, num_images):
  """Label IDs are extracted in synchronism with the data."""
  print('Extracting', filename)
  nparr = np.load(filename)
  nparr=np.int64(nparr)
  return nparr



def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def  makeTrainBatch(train_data, train_labels, BATCH_SIZE):
    '''Forms the training batch'''
    ind = np.random.randint(train_data.shape[0], size=[BATCH_SIZE])
    batch_data = train_data[ind,:,:,:]
    batch_labels = train_labels[ind]
    return (batch_data, batch_labels)
        


def main(argv=None):  # pylint: disable=unused-argument

  # Get the data.
  train_data_filename = ('./dest64/train_data.dat.npy')
  train_labels_filename = ('./dest64/train_label_val.dat.npy')
  test_data_filename = ('./dest64/test_data.dat.npy')
  test_labels_filename = ('./dest64/test_label_val.dat.npy')
  validation_data_filename = ('./dest64/valid_data.dat.npy')
  validation_labels_filename = ('./dest64/valid_label_val.dat.npy')

  # Extract it into numpy arrays.
  train_data = extract_data(train_data_filename, 35000)
  train_labels = extract_labels(train_labels_filename, 35000)
  test_data = extract_data(test_data_filename, 10000)
  test_labels = extract_labels(test_labels_filename, 10000)
  validation_data = extract_data(validation_data_filename, 5000)
  validation_labels = extract_labels(validation_labels_filename, 5000)

  train_data = np.vstack([train_data,validation_data,test_data])
  train_labels = np.hstack([train_labels,validation_labels,test_labels])

  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.1,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
#        print(predictions)
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(TRAIN_STEPS):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      
      (batch_data, batch_labels) = makeTrainBatch(train_data, train_labels, BATCH_SIZE)

      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph is should be fed to.
      print(step)
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)

      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess,"./saved_model.ckpt")


    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    saver = tf.train.Saver(tf.all_variables())
    saver.save(sess,"./saved_model_final.ckpt")
    print('training complete')

#
if __name__ == '__main__':
  tf.app.run()
#  


  





