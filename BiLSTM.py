"""
Introduction: 构建BiLSTM + CRF 模型进行中文分词
Author: gaochen3
Date: 2018.04.10
"""

import config
import numpy as np
import tensorflow as tf
from PrepareData import reader
from tensorflow.contrib import rnn

class BiLSTM_CRF:
    def __init__(self):
        """
        构造函数
        """
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.global_step = tf.Variable(-1, trainable=False, name='global_step')
        self.gStep = 0
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = False
        self.train_data = reader(config.train_file, config.dict_file, False)
        self.test_data = reader(config.test_file, config.dict_file, True)

    def weight_variable(self, shape):
        """
        权重初始化
        """
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

    def bias_variable(self, shape):
        """
        偏置初始化
        """
        return tf.Variable(tf.constant(0.1, dtype = tf.float32, shape = shape))


    def LSTM_Cell(self):
        """
        定义每一层LSTM单元结构，每一层中加入一层dropout
        """
        lstm_cell = rnn.BasicLSTMCell(config.hidden_size, reuse=tf.AUTO_REUSE)
        if config.train == True:
            lstm_cell = rnn.DropoutWrapper(lstm_cell, self.keep_prob)
        return lstm_cell


    def load_data(self):
        """
        加载数据, 返回训练集和测试集数据的迭代器
        """

        train_X = np.asarray(self.train_data.words_index, dtype = np.int32)
        train_y = np.asarray(self.train_data.labels_index, dtype = np.int32)

        test_X = np.asarray(self.test_data.words_index, dtype = np.int32)
        test_y = np.asarray(self.test_data.labels_index, dtype = np.int32)

        train_dataSet = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        train_dataSet = train_dataSet.batch(config.train_batch_size)

        test_dataSet = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        test_dataSet = test_dataSet.batch(config.test_batch_size)

        self.Iterrator = tf.data.Iterator.from_structure(train_dataSet.output_types, train_dataSet.output_shapes)

        self.train_Initializer = self.Iterrator.make_initializer(train_dataSet)
        self.test_Initializer = self.Iterrator.make_initializer(test_dataSet)

    def Build_model(self):
        """
        构建分词模型
        输入层: train_X [batch_size, time_step]
               embedding [batch_size, time_step, embedding_size]
        BI-LSTM: 多层LSTM构成, 采用动态序列长度
                 input: [batch_size, input_size]
                 output: (outputs, output_state)
        """
        with tf.variable_scope("inputs"):
            self.load_data()
            train_X, train_y = self.Iterrator.get_next()
            # 求出batch_size中每个序列的长度, seq_length: [batch_size]
            seq_length = tf.reduce_sum(tf.sign(train_X), 1)
            seq_length = tf.cast(seq_length, tf.int32)
            embedding = tf.Variable(tf.random_normal([config.vocab_size, config.embedding_size]), dtype = tf.float32)
            weights_out = tf.Variable(tf.random_normal([2 * config.hidden_size, config.class_num]))
            bais_out = tf.Variable(tf.random_normal([config.class_num]))
            inputs = tf.nn.embedding_lookup(embedding, train_X)
        with tf.variable_scope("biLSTM"):
            cell_fw = rnn.MultiRNNCell([self.LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
            cell_bw = rnn.MultiRNNCell([self.LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
            bilstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs = inputs, sequence_length = seq_length, dtype = tf.float32)
            output = tf.reshape(tf.concat(bilstm_output, 2), [-1, config.hidden_size*2])
        logits = tf.matmul(output, weights_out) + bais_out
        # CRF模型
        scores = tf.reshape(logits, [-1, config.max_sentence_len, config.class_num])
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, train_y, seq_length)
        self.loss = tf.reduce_mean(-log_likelihood)
        # 如果使用softmax则需要对loss做mask
        # losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(train_y, [-1]), logits = scores))
        # mask = tf.sequence_mask(seq_length)
        # losses = tf.boolean_mask(losses, mask
        # loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", self.loss)
        self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss, global_step = self.global_step)
        correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(train_y, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)


def train():
    """
    模型训练或者预测的主函数
    """
    model = BiLSTM_CRF()
    train_num = model.train_data.sample_nums
    test_num = model.test_data.sample_nums
    model.Build_model()
    saver = tf.train.Saver()
    with tf.Session(config = model.gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        if config.train == True:
            sess.run(model.train_Initializer)
            for epoch in range(config.epochs):
                tf.train.global_step(sess, global_step_tensor = model.global_step)
                for train_batch in range(train_num // config.train_batch_size):
                    _, batch_accuracy, batch_loss, summary, gStep  = sess.run([model.train_op, model.accuracy, model.loss, merged, model.global_step], feed_dict = {model.keep_prob : config.keep_prob})
                    # 加入tensorBoard可视化
                    if train_batch % config.per_summary == 0:
                        writer.add_summary(summary, epoch * (train_num // config.train_batch_size) + train_batch)
                    if train_batch % config.per_print == 0:
                        print("Epoch: {} Global Step: {} training loss: {} batch_accuracy: {}".format(epoch, gStep, batch_loss, batch_accuracy))
                if epoch % config.per_test == 0:
                    sess.run(model.test_Initializer)
                    for test_batch in range(test_num // config.test_batch_size):
                        print("Test Accuracy: {} step: {}".format(sess.run(model.accuracy, feed_dict = {model.keep_prob : config.keep_prob}), test_batch))
                # 每 3 个 epoch 保存一次模型
                if epoch % config.per_save == 0:
                    saver.save(sess, config.model_save_path, global_step = model.gStep)
        else:
            # 加载模型
            ckpt = tf.train.get_checkpoint_state(config.model_ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                print("Load model: {}".format(ckpt.model_checkpoint_path))
                # saver.restore(sess, ckpt.model_checkpoint_path)
            test_data = reader(config.test_file, config.dict_file, True)
            test_X, _ = test_data.get_batch(2)
            for word in test_X:
                test_data.index_str(word)


if __name__ == "__main__":
    train()
