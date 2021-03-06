import numpy as np
import tensorflow as tf

# Creating some things
a = tf.constant(2)
b = tf.constant(5)
c = tf.constant(10)

# Now some operations (nodes in the graph)
d = tf.add(a, b)
e = tf.multiply(a, b)

# But we can also use:
f = a + c
g = e * c

# Time to run the graph
sess = tf.Session()
eval_f = sess.run(f)
eval_g = sess.run(g)
sess.close()

print('f = {fval}\ng = {gval}'.format(fval = eval_f, gval = eval_g))
