import numpy as np
import tensorflow as tf

#x = np.random.uniform(0,1,(20,2))
#np.save('./samples/random_sample.npy', x)

"""
If all of your input data fit in memory, the simplest way to create a Dataset
from them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices().
"""
data = np.load('./samples/random_sample.npy')
dataset = tf.data.Dataset.from_tensor_slices(data)
#Note that the above code snippet will embed the arrays in your TensorFlow graph as tf.constant() operations

"""As an alternative, you can define the Dataset in terms of tf.placeholder()
 tensors, and feed the NumPy arrays when you initialize an Iterator over the dataset."""

data_placeholder = tf.placeholder(data.dtype, data.shape)
dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
iterator = dataset.make_initializable_iterator()
element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={data_placeholder: data})
    print(sess.run(element))
