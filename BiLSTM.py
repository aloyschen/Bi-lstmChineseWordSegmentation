"""
Introduction: 构建双向LSTM模型进行中文分词
Author: gaochen3
Date: 2018.04.10
"""

import config
from PrepareData import reader
import tensorflow as tf
from tensorflow.contrib import rnn

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, dtype = tf.float32, shape = shape))


def LSTM_Cell():
    cell = rnn.BasicLSTMCell(config.hidden_size, forget_bias = 0.1)
    dropout = rnn.DropoutWrapper(cell, config.keep_prob)
    return dropout


def BiLSTM(train_X, train_y):
    """

    :return:
    """
    with tf.variable_scope("inputs"):
        embedding = tf.get_variable("embedding", [config.vocab_size, config.embedding_size], dtype = tf.float32)
        weights_out = tf.Variable(tf.random_normal([2 * config.hidden_size, config.class_num]))
        bais_out = tf.Variable(tf.random_normal([config.class_num]))
        inputs = tf.nn.embedding_lookup(embedding, train_X)
    with tf.variable_scope("biLSTM"):
        LSTM_fw_cell = LSTM_Cell()
        LSTM_bw_cell = LSTM_Cell()
        cell_fw = rnn.MultiRNNCell([LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
        inputs = tf.unstack(inputs, config.time_step, 1)
        bilstm_output, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs = inputs, dtype = tf.float32)
        output = tf.reshape(bilstm_output, [-1, config.hidden_size*2])
    logits = tf.matmul(output, weights_out) + bais_out
    # y_pred = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(train_y, [-1]), logits = logits))
    tf.summary.scalar("loss", loss)
    # optimizer = tf.train.AdagradOptimizer(learning_rate = config.lr)
    # train_op = optimizer.minimize(loss)
    # ***** 优化求解 *******
    tvars = tf.trainable_variables()  # 获取模型的所有参数
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_grad_norm)   # 获取损失函数对于每个参数的梯度
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    # 梯度下降计算
    train_op = optimizer.apply_gradients(zip(grads, tvars),
                                         global_step=tf.contrib.framework.get_or_create_global_step())
    correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(train_y, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return train_op, accuracy, loss


def train():
    """
    :return:
    """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    input_x = tf.placeholder(tf.int32, [None, config.max_sentence_len], name = "input_X")
    input_y = tf.placeholder(tf.int32, [None, config.max_sentence_len], name = "input_y")
    train_op, accuracy, loss = BiLSTM(input_x, input_y)
    saver = tf.train.Saver()  # 最多保存的模型数量
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        data = reader(config.input_file, config.dict_file, config.input_dict)
        for epoch in range(config.epochs):
            for batch in range(data.sample_nums // config.batch_size):
                batch_X, batch_y = data.get_batch(config.batch_size)
                feed_dict = {input_x : batch_X, input_y : batch_y}
                _, batch_accuracy, batch_loss = sess.run([train_op, accuracy, loss], feed_dict = feed_dict)
                if batch % 100 == 0:
                    print("Epoch: {} iter_num: {} training loss: {} batch_accuracy: {}".format(epoch, batch, batch_loss, batch_accuracy))
            if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
                saver.save(sess, config.model_save_path, global_step=(epoch + 1))

if __name__ == "__main__":
    train()











