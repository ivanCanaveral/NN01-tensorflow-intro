import numpy as np
import tensorflow as tf

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
    initial_weights = tf.truncated_normal(1.0, shape=shape)
    return tf.Variable(initial_weights)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[-1]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return matmul(input, W) + b
