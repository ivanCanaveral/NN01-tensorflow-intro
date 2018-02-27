import tensorflow as tf
import numpy as np

"""
Using interactive session to do some math
"""

# We're going to do some matrix stuff
A = tf.constant([[1,2,3], [4,5,6]], name='A-matrix', dtype = tf.float64)
print('Shape of matrix A: {}'.format(A.get_shape()))

b = tf.constant([1,0,1], name='b-matrix', dtype=tf.float64)
print('Shape of matrix b: {}'.format(b.get_shape()))

# we need to make sure both have the same number of dimensions and that they
# are aligned correctly with respect to the intended multiplication
# transforming it from a 1D vector to a 2D single-column matrix
# We can add another dimension by passing the Tensor to tf.expand_dims() , together
# with the position of the added dimension as the second argument. By adding another
# dimension in the second position (index 1), we get the desired outcome

b = tf.expand_dims(b, 1)
print('New shape of matrix b: {shape}'.format(shape=b.get_shape()))

""" can I do this??
with tf.InteractiveSession() as sess:

thows the following error: AttributeError: __exit__

probably not"""
sess = tf.InteractiveSession()

# We can still define nodes once the session as been initialized.
# ¿Only with InteractiveSession?
c = tf.matmul(A,b)

# Using interactive sessions we don't have to use sess.run(fetches) when
# executing things. With tensor.eval() is enough
print(c.eval())

sess.close()
