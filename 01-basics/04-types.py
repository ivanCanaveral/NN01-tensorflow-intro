import tensorflow as tf

print("Simple constant")
c = tf.constant(4, dtype=tf.float64)
print(c)
print(c.dtype)

print("\nNamed constant")
d = tf.constant([1,1,1], name='my-constant', dtype=tf.int64)
print(d)
print(d.dtype)
print("Casting the constant")
e = tf.cast(d, tf.float64)
print(e.dtype)

print("\nNow with np.arrays")
import numpy as np
a = np.random.normal(0,1,(10,5))
c = tf.constant(a, name='from-np-array', dtype=tf.float16)
print(c)
