import numpy as np
import tensorflow as tf

print('[Dataset]')

print('[Dataset][batch]')
print('[Dataset][batch] The default value is one. We can change it')
"""
we can use the method batch(BATCH_SIZE) that automatically batches the dataset
with the provided size. The default value is one.
"""
BATCH_SIZE = 5
data = np.random.sample((100,2))
ds = tf.data.Dataset.from_tensor_slices(data).batch(BATCH_SIZE)

itr = ds.make_one_shot_iterator()
element = itr.get_next()

with tf.Session() as sess:
    print(sess.run(element))

print('\n')
print('[Dataset][repeat]')
print('[Dataset][repeat] we can specify the number of times we want the dataset to be iterated')
"""
Using .repeat() we can specify the number of times we want the dataset to be
iterated. If no parameter is passed it will loop forever, usually is good to
just loop forever and directly control the number of epochs with a standard loop.
"""
N_REPS = 3
data = np.random.sample((5,2))
ds = tf.data.Dataset.from_tensor_slices(data).repeat(N_REPS)

itr = ds.make_one_shot_iterator()
element = itr.get_next()

with tf.Session() as sess:
    for _ in range(15):
        print(sess.run(element))

print('\n')
print('[Dataset][shuffle]')
print('[Dataset][shuffle] We can shuffle the Dataset')
BATCH_SIZE = 10
N_REPS = 3
data = np.arange(1,100)
# todo son atributos del dataset, así que no haye falta hacerlo todo del tirón
# ¡Ojito! Que por eso mismo importa el orden, no es lomismo batch.suffle que suffle.batch
# uno mezcla los números y hace paquetes, y otro hace paquetes y luego los mezcla
ds = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(buffer_size=5).batch(BATCH_SIZE)
# buffer size controls teh window from wich each element can be shuffled

itr = ds.make_one_shot_iterator()
element = itr.get_next()

with tf.Session() as sess:
    for _ in range(15):
        print(sess.run(element))

print('\n')
print('[Dataset][example]')
print('[Dataset][example] a simple nn example')

EPOCHS = 10
BATCH_SIZE = 10

features= np.random.sample((100,2))
labels= np.random.sample((100,1))
# a form_tensor_slices hay que darle una única cossa. O un array,
# o una tupla de arrays
ds = tf.data.Dataset.from_tensor_slices((features, labels)).repeat().batch(BATCH_SIZE)
itr = ds.make_one_shot_iterator()
x, y = itr.get_next() # como hemos construido el dataset con una tupla, esto
                      # devuelve tuplitas

# network
net = tf.layers.dense(x, 8, activation= tf.tanh)
net = tf.layers.dense(net, 8, activation= tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, y)
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Step {i}, loss: {l:.4f}".format(i=i, l=loss_value))
