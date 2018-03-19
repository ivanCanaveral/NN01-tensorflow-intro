import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial_weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_weights)
