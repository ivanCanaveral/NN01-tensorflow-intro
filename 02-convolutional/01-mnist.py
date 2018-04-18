import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './samples'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

def weight_variable(shape):
    """
        tf.truncated_normal(
        shape,
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
        seed=None,
        name=None
    )"""
    initial_weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_weights)

def bias_variable(shape):
    initial_weights = tf.constant(1.0, shape=shape)
    return tf.Variable(initial_weights)

def conv2d(x, W):
    """ Adds a full convolutive layer """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

def conv_layer(input, shape):
    """ Create the weights and mount the layer

    shape is the shape of the weights"""
    W = weight_variable(shape)
    b = bias_variable([shape[3]]) #as many bias as filters
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# -1 means that the length in that dimension is inferred
x_images = tf.reshape(x, [-1, 28, 28, 1])

# First convolution.
#   32 filters of 5x5.
#   inputs:  [-1, 28, 28, 1]
#   weights: [5, 5, 1, 32]
#   output:  [-1, 28, 28, 32] Esto tiene que ver con los movimientos que pueden hacer los filtros
conv1 = conv_layer(x_images, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

# Second convolution
#   64 filters of 5x5x32
#   weights:
#   output:
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

# mnist
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

print('Labels shape: ', y_.get_shape())
print('Predictions shape:', y_conv.get_shape())

# error
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))

train_step = tf.train.AdamOptimizer(1e-04).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(NUM_STEPS):
        batch = mnist.train.next_batch(50)
        #print(batch[0])
        #print(batch[0])
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0
            })

            print('step {}, trainning accuracy {} %'.format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    batch = mnist.train.next_batch(1)
    #print(batch)
    print(batch[1])
    print(sess.run(y_conv, feed_dict={x:batch[0], keep_prob: 1}))

    batch = mnist.train.next_batch(1)
    print(batch[1])
    print(sess.run(y_conv, feed_dict={x:batch[0], keep_prob: 1}))

    X = mnist.test.images.reshape(10,1000,784)
    Y = mnist.test.labels.reshape(10,1000,10)
    test_accuracy = np.mean([sess.run(accuracy,feed_dict={x: X[i], y_: Y[i], keep_prob:1.0}) for i in range(10)])

print('test accuracy: {}'.format(test_accuracy))
