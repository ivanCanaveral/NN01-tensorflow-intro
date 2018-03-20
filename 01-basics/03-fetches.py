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

with tf.Session() as sess:
    fetches = [a,b,c,d,e,f,g]
    outs = sess.run(fetches)
    print("fetches = {}".format(outs))
