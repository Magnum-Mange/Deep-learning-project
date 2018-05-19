#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os;
import nibabel as nib;
import time;
import math;
import random;
from tensorflow.python.client import device_lib
print(tf.__version__);
print(device_lib.list_local_devices());


tf.logging.set_verbosity(tf.logging.INFO)

global BATCH_SIZE;
BATCH_SIZE = 1;


def lossFunc(predictions, groundTruth):
  #convert predictions and groundTruth to binary rep
  # n = len(predictions);
  # y = tf.cast(tf.range(n), tf.int64);
  #tried to cast x to int32 even though groundTruth was cast to int32 before function call
  for i in range(tf.shape(predictions)):
    # predictions[i] = tf.mod(tf.bitwise.right_shift(tf.expand_dims(predictions[i], 1), tf.range(n)), 2);
    predictions[i] = int(bin(predictions[i])[2:]);

  # n = len(groundTruth);
  # y = tf.cast(tf.range(n), tf.int64);
  for i in range(tf.shape(groundTruth)):
    groundTruth[i] = int(bin(groundTruth[i])[2:]);

  eps = 10 ** -8;
  numerator = tf.reduce_sum(predictions * groundTruth);
  denom = eps + tf.reduce_sum(predictions) + tf.reduce_sum(groundTruth);

  return 2 * numerator / denom;

#INPUT FUNCTION
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 240, 240, 155, 1])
  print(input_layer.shape)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 240, 240, 155, 1]
  # Output Tensor Shape: [batch_size, 240, 240, 155, 32]
  conv1 = tf.layers.conv3d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5, 5],
      padding="same",
      activation=tf.nn.relu)
  print(conv1.shape);
  print("after conv 1");
  print(conv1);
#   !free -g
  # print(conv1);
  # print(type(conv1))
  # print(tf.cast(conv1, tf.float32))
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 240, 240, 155, 32]
  # Output Tensor Shape: [batch_size, 120, 120, 155, 32]
  pool1 = tf.layers.max_pooling3d(inputs=tf.cast(conv1, tf.float32), pool_size=[2, 2, 2], strides=2)

  print("after pool 1");
  print(pool1);
#   !free -g
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 120, 120, 155, 32]
  # Output Tensor Shape: [batch_size, 120, 120, 155, 64]
  conv2 = tf.layers.conv3d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5, 5],
      padding="same",
      activation=tf.nn.relu)

  print("After conv 2");
  print(conv2);
#   !free -g
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 120, 120, 155*32*2]
  # Output Tensor Shape: [batch_size, 60, 60, 64]
  pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
  print("after pool 2")
  pool2 = tf.cast(pool2, tf.float32);
  print(pool2)
#   !free -g
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 60, 60, 64]
  # Output Tensor Shape: [batch_size, 60* 60* 64]


  #DOES NOT LIKE THIS - TOO BIG?
  pool2_flat = tf.reshape(pool2, [-1, 60 * 60 * 38*32*2]);#batch_size == 5 here
  print("After pool 2 flat");
  print(pool2_flat)
#   !free -g
  
  #DOES NOT LIKE THIS!!!!
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 60 * 60 * 155*32*2]
  # Output Tensor Shape: [batch_size, 256]
  dense = tf.layers.dense(inputs=pool2_flat, units=64, activation=tf.nn.relu) #changed units from 1024 to 256
  print("after dense");
  print(dense);
#   !free -g

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  print("After dropout");
  print(dropout);
#   !free -g





  # Logits layer
  # Input Tensor Shape: [batch_size, 60 * 60 * 155*32*2]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)
  print("After logits");
  print(logits);


  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.cast(tf.nn.softmax(logits, name="softmax_tensor"), tf.int32)
  }

  print("after predictions");
  print(predictions);
  if mode == tf.estimator.ModeKeys.PREDICT:
#     !free -g
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)




  print("\n\n\n\n\n")
  prob = tf.cast(tf.nn.softmax(logits, name="softmax_tensor"), tf.int32);
  print(prob);
  # prob = prob.eval();


  # Calculate Loss (for both TRAIN and EVAL modes)
  # print(type(logits));
  # print("\n\n\n\n")
  # print(logits);
  # print(type(logits))
  # print(labels)
  # print(labels.shape)


  labels = tf.cast(labels, tf.int32);
  # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  #DICE LOSS FUNCTION
  loss = lossFunc(prob, labels);
  print("After loss");

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    # print("1");
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    # print("2");
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  print("after TRAIN mode check");
  
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  print("eval_metric_ops");
#   !free -g
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  


def load_image_data(path):
  #returns train_data, eval_data, train_labels, eval_labels
  numFolders = len(os.listdir(path)); #MAY HAVE OFF BY 1 ISSUE HERE
  # trainingSize = math.floor(numFolders * .8);
  # evalSize = numFolders - trainingSize;
  numFolders = 10;
  trainingSize = 5;
  evalSize = 5;
  print(numFolders);
#   print(numFolders);
  retVal = np.zeros((trainingSize, 240 * 240 * 155), dtype='float32');#5 works - should be numFolders
  train_labels = np.zeros((trainingSize, 240 * 240 *155), dtype = 'float32');

  eval_data = np.zeros((evalSize, 240 * 240 * 155), dtype='float32');
  eval_labels = np.zeros((evalSize, 240 * 240 * 155), dtype='float32');
  allBrains = os.listdir(path);
  random.shuffle(allBrains);

#   print(type(retVal[1, 1]))
  #WITH all 5 images, we want the above to be numFolders x 5 x 240 x 240 x 155
  curFolder = 0;
#   print("Before outer for");
  for folder in allBrains:
#     print(os.listdir(path));
    if folder == ".DS_Store":
      continue;
    folderpath = path + "/" + folder;
#     print(folderpath)

    curImg = 0;
    
    allFiles = (os.listdir(folderpath));

    for filename in sorted(allFiles):
      if "t1ce" not in filename or "seg" not in filename:
        continue;
      file = os.path.join(folderpath, filename);
      img = nib.load(file);
      img_data = img.get_data();
      print(type(img_data));
      if curFolder < trainingSize:
        if "t1ce" in filename:
          retVal[curFolder] = img_data.reshape((1, 240 * 240 * 155));
        else:
          #We want the img_data to be a one-hot representation of where the tumor is in the brain
          train_labels[curFolder] = img_data.reshape((1, 240 * 240 * 155));
      else:
        if "t1ce" in filename:
          eval_data[curFolder] = img_data.reshape((1, 240 * 240 * 155));
        else:
          #We want the img_data to be a one-hot representation of where the tumor is in the brain
          #convert to binary
          eval_labels[curFolder] = img_data.reshape((1, 240 * 240 * 155));
      if curFolder == numFolders:
        break;
      # retVal[curFolder][curImg] = img_data;
      # curImg += 1;
    curFolder += 1;
    print(str(float(curFolder) / float(numFolders) * 100) + "% done");

  return retVal, eval_data, train_labels, eval_data;







def main(unused_argv):
  # Load training and eval data
#   mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
  '''
    STEP 1 - Load Data (training and validation)
  '''
  # train_data = mnist.train.images  # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # print(train_data.shape)

  # tmp = [np.where(r == 1)[0][0] for r in train_labels];
  # train_labels = tmp;
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  # tmp = [np.where(r == 1)[0][0] for r in eval_labels];
  # eval_labels = tmp;

  print("LGG")
  train_data, eval_data, train_labels, eval_labels = load_image_data("Folder");
  # train_labels = np.zeros((train_data.shape[0]));
  # for i in range(train_data.shape[0]):
  #   train_labels[i] = 1;
  # print("HGG");
  # eval_data = load_image_data("Folder/HGG");
  # print(eval_data.shape);
  # try:
  #   eval_labels = np.zeros((eval_data.shape[0]));
  #   for i in range(eval_data.shape[0]):
  #     eval_labels[i] = 1;
  # except:
  #   print("something")


  train_labels = np.asarray((train_labels))
  eval_labels = np.asarray((eval_labels));

  '''
  STEP 2 - Create estimator?
  '''
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model3")
  print("After estimator");
  
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}#dictionary
  # tensors_to_log = np.array(list(tensors_to_log.items()))
  # print(tensors_to_log.shape);
  #tensors_to_log must be a dictionary to be passed into LoggingTensorHook
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)#every_n_iter = 50
  print("After logging hook");


  '''
  STEP 3 - Train Model
  '''
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True);
  print("After train model");

  mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook], steps=1);
  print("Before evaluating model and print results");  
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  print("after estimator inputs numpy");
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
