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
### MAP
benchmark_data_a = np.random.sample((1000000,10))
benchmark_data_b = np.random.sample((1000000,10))
benchmark_dataset = tf.data.Dataset.from_tensor_slices((benchmark_data_a, benchmark_data_b))

for i in range(4):
    t0 = time()
    b = benchmark_dataset.map(lambda x, y: ((x**2)+ x/2, y**5), num_parallel_calls=i*10)
    itr = b.make_one_shot_iterator()
    e = itr.get_next()
    with tf.Session() as sess:
        for j in range(1000000):
            a = e
    print("time spend with {i} threads: {t} s".format(i=i, t=time() - t0))
