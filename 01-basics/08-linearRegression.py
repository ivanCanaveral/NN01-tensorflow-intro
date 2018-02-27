import tensorflow as tf
import numpy as np
from time import time

# Data simulation
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real,x_data.T) + b_real + noise


NUM_STEPS = 20

g = tf.Graph()

with g.as_default():

    x = tf.placeholder(tf.float32, shape=(None, 3))
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('init') as scope:
        # dos paréntesis para que tenga dimensión dos y poder hacer matmul
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name='weigths')
        b = tf.Variable(0, dtype=tf.float32, name='bias')

    with tf.name_scope('definitions'):
        y_pred = tf.matmul(w, tf.transpose(x)) + b
        loss = tf.reduce_mean(tf.square(y_pred - y_true))

    with tf.name_scope('minimization-algorithm'):
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimization_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            t0 = time()
            sess.run(optimization_step, feed_dict={x: x_data, y_true: y_data})
            time_spent = time() - t0
            print('Time spent in {step} step: {time}'.format(step=step, time=time_spent))
            if (step % 5) == 0:
                # Sospecho que cuando hacemos este run w y b no se vuelven a ejecutar
                # si no que se quedan almacenados los valores
                t0 = time()
                w_value, b_value = sess.run([w,b]) # parece que esto no ejecuta, sólo lee
                time_spent = time() - t0
                print('\nStep {step}, time_to_read: {time} \nw: {w}, \nb: {b}'.format(
                    step=step, time=time_spent, w=w_value, b=b_value))
                # Si queremos ver como anda loss, no podemos hacer simplemente
                # sess.run(loss), porque esto si que ejecuta cosas, y necesita el feed_dict
                # Lo otro que son variables, no.
                """ Ojito
                No podemos hacer:

                loss_value = sess.run(loss, feed_dict={w:w, b:b, y_true:y_true})

                porque nos dice: The value of a feed cannot be a tf.Tensor object.
                Acceptable feed values include Python scalars, strings, lists,
                numpy ndarrays, or TensorHandles.

                Tenemos que poner los que acabamos de sacar. Aunque esto... volverá
                a ejecutar cosas. """
                t0 = time()
                loss_value = sess.run(loss, feed_dict={w:w_value,
                    b:b_value, x:x_data, y_true:y_data}) # esto vuelve a ejecutar!!
                time_spent = time() - t0
                print('loss: {loss}, time: {time}'.format(loss=loss_value,
                    time=time_spent))


        print(w, b) # Esto imprime las aristas del grafo
        print(sess.run([w, b])) # Esto imprime los valores
