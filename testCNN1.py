import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import prettytable as pt
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import model_analyzer
import linecache


tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]


##
tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y


# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)


pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)

conv2 = tf.layers.conv2d(   # shape (14, 14, 16)
    inputs=pool1,
    filters=32,
    kernel_size=3,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (14, 14, 32)


pool2 = tf.layers.max_pooling2d(
    conv2,
    pool_size=2,
    strides=2,
)           # -> (7, 7, 32)

conv3 = tf.layers.conv2d(   # shape (7, 7, 32)
    inputs=pool2,
    filters=64,
    kernel_size=2,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (7, 7, 64)

pool3 = tf.layers.max_pooling2d(
    conv3,
    pool_size=7,
    strides=7,
)           # -> (1, 1, 64)

flat = tf.reshape(pool3, [-1, 1*1*64])          # -> (7*7*32, ), -1 is inferred to be 1
output = tf.layers.dense(flat, 10)              # output layer

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph



########################################################################################################__________________________________________________
import script
image_for_structure={tf_x: test_x[:1]}
script.func(sess,output,image_for_structure)
#script.func(output,image_for_structure)
########################################################################################################__________________________________________________



# training
for step in range(50):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)



