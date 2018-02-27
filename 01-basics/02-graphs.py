import tensorflow as tf

# As soon as we import tensorflow, a default graph is created
print('The default graph: {g}'.format(g=tf.get_default_graph()))

g = tf.get_default_graph()

h = tf.Graph()
print('Other new graph: {h}'.format(h=h))
print('Creating node a...')
a = tf.constant(8)
print('Is a in the new graph? {}'.format(a.graph is h))
print('Is a in the default graph? {}'.format(a.graph is g))

print('Is the new graph the active graph? {}'.format(g is tf.get_default_graph()))

# Now using a context manager
with h.as_default():
    print('Is the new graph the active graph? {}'.format(g is tf.get_default_graph()))

# Once the context manager is closed
print('Is the new graph the active graph? {}'.format(g is tf.get_default_graph()))
