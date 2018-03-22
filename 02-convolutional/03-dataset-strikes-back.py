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

dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)
"""
### map
x = np.random.sample((100,2))
ds = tf.data.Dataset.from_tensor_slices(x)
ds = ds.map(lambda x: x + 10, num_parallel_calls=None)
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

"""
Extracting values from iterators

If the iterator reaches the end of the dataset, executing the Iterator.get_next()
operation will raise a tf.errors.OutOfRangeError. After this point the iterator
will be in an unusable state, and you must initialize it again if you want to
use it further.
"""
print('[How to extract tensors from a iterator]')

## Range Creates a Dataset of a step-separated range of values.
#Dataset.range(5) == [0, 1, 2, 3, 4]
#Dataset.range(2, 5) == [2, 3, 4]
#Dataset.range(1, 5, 2) == [1, 3]
#Dataset.range(1, 5, -2) == []
#Dataset.range(5, 1) == []
#Dataset.range(5, 1, -2) == [5, 3]
## ------
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer) # Si se crea desde un placeholder, necesitará feed_dict
                                   # en otro caso no.
    for _ in range(8):
        try:
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            print('Sacabó!')

    # if we want to reuse the iterator, we have to initialise it again!
    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            print('Sacabó!!')
            break

""" (!) Iterator that returns several tensors """
dataset1 = tf.data.Dataset.from_tensor_slices(tf.range(5))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.range(5), tf.range(5)))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

element1, (element2, element3) = iterator.get_next()

# Note that evaluating any of next1, next2, or next3 will advance the iterator
# for all components. A typical consumer of an iterator will include all
# components in a single expression.
with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run(element1))
    print(sess.run(element1))
    print(sess.run([element1, element2, element3]))
