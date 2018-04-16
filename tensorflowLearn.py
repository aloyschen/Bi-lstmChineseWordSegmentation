import numpy as np
import tensorflow as tf

t1 = tf.Variable(tf.random_normal([2,3,2]))
t3 = tf.unstack(t1, axis=0)
t2 = tf.concat([t1, t3], -1)
embedding = tf.get_variable("name", [3, 1], dtype = np.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
# sess.run(tf.global_variables_initializer())
    print(sess.run(t1))
    print(sess.run(t3))
    print(sess.run(tf.stack(t3, axis = 1)))
    # print(sess.run(t2))
