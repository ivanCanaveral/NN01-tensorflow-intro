# MNIST & Datasets

Here `mnist` is an object of type `Datasets`, which contains three objects of type `Dataset`: train, validation, test.

## Dataset,

The tf.data module contains a collection of classes that allows you to easily load data, manipulate it, and pipe it into your model.

TensoFlow has several API layers, that we can see here:

![TensorFlow API layers](https://www.tensorflow.org/images/tensorflow_programming_environment.png)

  * Estimators are the high-level API
  * Layers, datasets and metrics are the mid-level API
  * Python, C++, Go, and Java have their low-level API interfaces
  * Under the low-level APIs, we have the Kernel of tensorflow, known as _tensorflow distributed execution engine_

To use the dataset pipeline we should import data and create an iterator.

## Iterators

There are four types of iterators, one shot, initializable, reinitializable and feedable.

If the iterator reaches the end of the dataset, executing the `Iterator.get_next()`
operation will raise a `tf.errors.OutOfRangeError`. After this point the iterator
will be in an unusable state, and you must initialize it again if you want to use
it further.
