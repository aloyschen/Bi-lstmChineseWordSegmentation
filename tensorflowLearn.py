import numpy as np
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt


t1 = tf.constant([[574, 912, 574, 912, 107],
 [151, 273,   0,   0,   0]])
# t2 = tf.concat([t1, t3], -1)
embedding = tf.get_variable("name", [3, 1], dtype = np.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(t1))
    print(sess.run(tf.sign(t1)))
    print(sess.run(tf.reduce_sum(tf.sign(t1), axis=1)))
# with open('./data/pku_training.utf8', encoding = 'utf-8') as file:
#     line = file.read()
#     result = ''
#     data = line.splitlines()
#     for sentence in data:
#         words = sentence.split("  ")
#         for word in words:
#             if len(word) == 0:
#                 continue
#             if len(word) == 1:
#                 result += word + '/s '
#             elif len(word) > 2:
#                 length = len(word)
#                 result += word[0] + '/b '
#                 for i in range(1, length - 1):
#                     result += word[i] + '/m '
#                 result += word[-1] + '/e '
#             else:
#                 result += word[0] +'/b '
#                 result += word[-1] + '/e '
#     with open('./data/pku_data.txt', 'w+', encoding = 'utf-8') as file:
#         file.write(result)
#     print(result)

# image = img.imread('./test1.jpg')
# with tf.Session() as sess:
#     shape = tf.shape(image).eval()
#     h = shape[0]
#     w = shape[1]
#     standardization_image = tf.image.per_image_standardization(image)
#     fig = plt.figure()
#     ax = fig.add_subplot(311)
#     ax.imshow(image)
#     ax = fig.add_subplot(312)
#     ax.hist(sess.run(tf.reshape(image, [h * w, -1])))
#     ax = fig.add_subplot(313)
#     ax.hist(sess.run(tf.reshape(standardization_image,[h*w,-1])))
#     plt.show()
# ax_std.imshow(standardization_image)
