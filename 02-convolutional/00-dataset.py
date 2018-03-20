import tensorflow as tf
import numpy as np
import pandas as pd

dataframe = pd.read_csv('../data/iris_test.csv')

# to create a dataset we need features (values), labels, and a batch size
# lets extract all the values from a dataset of pandas
# and generate a Dataset of tensorflow from tensor slices,
# that stores each element of the np.array of the values
iris_ds = tf.data.Dataset.from_tensor_slices(dataframe.values)
print('Only with values, we get a set of elements: \n', iris_ds)
# >>> <TensorSliceDataset shapes: (5,), types: tf.float64>
# collection of arrays of len 5

#Datasets transparently handle any nested combination of dictionaries or tuples.
# For example, ensuring that features is a standard dictionary, you can then
#convert the dictionary of arrays to a Dataset of dictionaries as follows:

"""
feed-dict is the slowest possible way to pass information to TensorFlow and it must be avoided.
"""
# Creating it from numpy array
features = np.random.sample((100,2))
dataset = tf.data.Dataset.from_tensor_slices(features)
print('From numpy array: \n', dataset)
# with labels too
labels = np.random.randint(0,100,(100,1))
ds = tf.data.Dataset.from_tensor_slices((features, labels))
print('From numpy array with labels: \n', ds)

# Creating it from tensor
ds = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100,2]))
print('From a tensor object: \n', ds)

# From a placeholder too
x = tf.placeholder(tf.float32, shape=[None, 2])
ds = tf.data.Dataset.from_tensor_slices(x)
print('From a placeholder: \n', ds)

# Also using a generator, useful when we have elements of diferent sizes
def gen():
    seq = np.array([[1],[2,3],[4,5]])
    for element in seq:
        yield element
# In this case you also need specify the types and the shapes of your data that
# will be used to create the correct tensors.
ds = tf.data.Dataset.from_generator(gen, output_types=tf.float32, output_shapes=[tf.float32])
print('From a generator: \n', ds)

# Iterators
#   we have four types of iterators:
print('\n__Iterators__')
#    - one_shot_iterator
print('__one_shot_iterator')
features = np.random.sample((100,2))
ds = tf.data.Dataset.from_tensor_slices(features)
itr = ds.make_one_shot_iterator()
e = itr.get_next()
with tf.Session() as sess:
    print(sess.run(e))

print('__initializable_iterator') # useful for use the same iterator in the train
# an after that, for the test
x = tf.placeholder(tf.float32, shape=[None, 2])
ds = tf.data.Dataset.from_tensor_slices(x)

itr = dataset.make_initializable_iterator()
e = itr.get_next()

with tf.Session() as sess:
    sess.run(itr.initializer, feed_dict={x: features}) #this time we need to initialize
    # this in order to get data
    print(sess.run(e))

""" example
# initializable iterator to switch between dataset
EPOCHS = 10
x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))
iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
with tf.Session() as sess:
#     initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1]})
    for _ in range(EPOCHS):
        sess.run([features, labels])
#     switch to test data
    sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1]})
    print(sess.run([features, labels]))
"""

print('__reinitiable_iterator') # The concept is similar to before, we want to
# dynamic switch between data. But instead of feed new data to the same dataset, we switch dataset.
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))
# creating both datasets
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
features, labels = iter.get_next()
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)

EPOCHS = 5
with tf.Session() as sess:
    sess.run(train_init_op) # switch to train dataset
    for _ in range(EPOCHS):
        sess.run([features, labels])
    sess.run(test_init_op) # switch to val dataset
    print(sess.run([features, labels]))

print('__feedable_iterator')
#Honestly, I don’t think they are useful. Basically instead of switch between
#datasets they switch between iterators so you can have, for example, one iterator
# from make_one_shot_iterator() and one from ` make_initializable_iterator().
print('ñeh')

"""
to continue:
https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
https://www.tensorflow.org/programmers_guide/datasets#basic_mechanics
https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
"""
