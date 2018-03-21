import numpy as np
import tensorflow as tf
from time import time

# https://www.tensorflow.org/programmers_guide/datasets
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# https://www.tensorflow.org/versions/master/performance/datasets_performance

"""
We know how to do this:
"""
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print('dataset1, types: ', dataset1.output_types)  # ==> "tf.float32"
print('dataset1, shapes: ', dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print('dataset2: ', dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print('dataset2: ', dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print('dataset3, types: ', dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print('dataset3, types: ', dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

"""
It is often convenient to give names to each component of an element,
for example if they represent different features of a training example.
In addition to tuples, you can use collections.namedtuple or a dictionary
mapping strings to tensors to represent a single element of a Dataset.

vamos, que se pueden crear las cosas con tuplas o con diccinarios
"""
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

"""
map, flat_map & filter
"""
### map
x = np.random.sample((100,2))
ds = tf.data.Dataset.from_tensor_slices(x)
ds = ds.map(lambda x: x + 10)
itr = ds.make_one_shot_iterator()
e = itr.get_next()

with tf.Session() as sess:
    for _ in range(10):
        print(sess.run(e))

### filter
x = np.random.sample((100, 2))
ds = tf.data.Dataset.from_tensor_slices(x)
ds = ds.filter(lambda x: x[0] < 0.5)
itr = ds.make_one_shot_iterator()
e = itr.get_next()

with tf.Session() as sess:
    for _ in range(10):
        print(sess.run(e))
