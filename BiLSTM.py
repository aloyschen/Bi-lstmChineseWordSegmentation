"""
Introduction: 构建双向LSTM模型进行中文分词
Author: gaochen3
Date: 2018.04.10
"""

import config
from PrepareData import PreProcessing
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
    embedding = tf.get_variable("embedding", [config.vocab_size, config.embedding_size], dtype = tf.float32)
    weights_out = tf.Variable(tf.random_normal([2 * config.hidden_size, config.class_num]))
    bais_out = tf.Variable(tf.random_normal([config.class_num]))
    inputs = tf.nn.embedding_lookup(embedding, train_X)
    LSTM_fw_cell = LSTM_Cell()
    LSTM_bw_cell = LSTM_Cell()
    bilstm_output, bilstm_output_state= tf.nn.bidirectional_dynamic_rnn(LSTM_fw_cell, LSTM_bw_cell, inputs, dtype = tf.float32)
    output = tf.concat([bilstm_output[0], bilstm_output[1]], -1)
    logits = tf.matmul(output, weights_out) + bais_out
    y_pred = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = train_y, logits = logits))
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdagradOptimizer(learning_rate = config.lr)
    train_op = optimizer.minimize(loss)
    correct_pred = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(train_y, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return train_op, accuracy


def train():
    """
    :return:
    """
    prepare_pku = PreProcessing(4000, './data/pku_training.utf8', './data/pku_training_words.txt',
                            './data/pku_training_labels.txt', './data/pku_training_dict.txt', './data/pku_training_raw.utf8')
    train_X = prepare_pku.words_index
    train_y = prepare_pku.labels_index
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())









