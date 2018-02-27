import tensorflow as tf
import numpy as np

"""
Objects residing within the same graph cannot have the same name
—TensorFlow forbids it. As a consequence, it will automatically
add an underscore and a number to distinguish the two.
"""

with tf.Graph().as_default():
    # Names are case-sensitive
    c1 = tf.constant(4,dtype=tf.float64,name='an_unique_name')
    c2 = tf.constant(4,dtype=tf.int32,name='an_unique_name')
print(c1.name)
print(c2.name)

"""
Name scopes
Sometimes when dealing with a large, complicated graph, we would like to create
some node grouping to make it easier to follow and manage."""

# (!) Scopes are part of the name
with tf.Graph().as_default():
    nodeA = tf.constant(1, name='name_without_scopes')
    with tf.name_scope('branch-A'):
        nodeB = tf.constant(1, name='name_with_scope')
        nodeC = tf.constant(1, name='name_with_scope')
    with tf.name_scope('branch-N'):
        nodeD = tf.constant(1, name='name_with_scope')
        with tf.name_scope('subscope'):
            nodeE = tf.constant(1, name='name_with_scope')

print(nodeA.name)
print(nodeB.name)
print(nodeC.name)
print(nodeD.name)
print(nodeE.name)
