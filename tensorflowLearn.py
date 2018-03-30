import tensorflow as tf
import numpy as np
def placeholder():
    """
    最简单的一个tensorflow实例
    """
    input_x = tf.placeholder(tf.float32, [None, 2])
    x = np.random.rand(12, 2)
    with tf.Session() as sess:
        print(sess.run(input_x, feed_dict={input_x : x}))
placeholder()