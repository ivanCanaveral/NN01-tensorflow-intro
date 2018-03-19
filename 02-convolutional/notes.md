
# Convolutional layers

The basic pieces of the convolutional neural networks.

## Layers

Some specific layers an the way they are used in

### conv2d

`tf.nn.conv2d(x, W, strides, padding)`

  * x: layer input `[None, nrows, ncolumns, nchannels]`
  * W: weights `[npixels_window_x, npixels_window_y, n_channels, nfeatures_map]`
        each filter is applied with the same weights in all the input.
  * strides: defines de movement of the window along all dimensions `[i, j, k, l]``
  * padding: _'SAME'_ moves the filters in such a way that the output has the same size as the input.

### pool

Reduces the output grouping in blocks of nxm. Here we show the max_pool function.

`tf.nn.max_pool(x, ksize, strides, padding)`

  * ksize: the size of the pooling. For example, a 2x2 pixel pooling is like `[1, 2, 2, 1]`
  * strides: the jump of the block. A good idea is to jump the same of units as the blocks size.
  * padding: the same as above

### dropout

Removing some nodes.

`tf.nn.dropout(layer, keep_prob=keep_prob)`

  * layer: a layer to dropout nodes from
  * keep_prob: probability of keep the node in the layer. 1.0 = no dropout

## docs

#### tf.truncated_normal

```python
tf.truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
)
```

#### tf.Variable

Creates a new variable with value initial_value.

The new variable is added to the graph collections listed in `collections`, which defaults to `[GraphKeys.GLOBAL_VARIABLES]`.

If `trainable` is `True` the variable is also added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.

This constructor creates both a `variable` Op and an `assign` Op to set the variable to its initial value.

```python

__init__(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None,
    constraint=None
)
´´´
