import numpy as np
import tensorflow as tf

# t1 = tf.Variable(tf.random_normal([2,3,2]))
# t3 = tf.unstack(t1, axis=0)
# t2 = tf.concat([t1, t3], -1)
# embedding = tf.get_variable("name", [3, 1], dtype = np.int32)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
# # sess.run(tf.global_variables_initializer())
#     print(sess.run(t1))
#     print(sess.run(t3))
#     print(sess.run(tf.stack(t3, axis = 1)))
#     # print(sess.run(t2))
with open('./data/pku_training.utf8', encoding = 'utf-8') as file:
    line = file.read()
    result = ''
    data = line.splitlines()
    for sentence in data:
        words = sentence.split("  ")
        for word in words:
            if len(word) == 0:
                continue
            if len(word) == 1:
                result += word + '/s '
            elif len(word) > 2:
                length = len(word)
                result += word[0] + '/b '
                for i in range(1, length - 1):
                    result += word[i] + '/m '
                result += word[-1] + '/e '
            else:
                result += word[0] +'/b '
                result += word[-1] + '/e '
    with open('./data/pku_data.txt', 'w+', encoding = 'utf-8') as file:
        file.write(result)
    print(result)

