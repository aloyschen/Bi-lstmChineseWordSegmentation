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
        self.train_data = reader(input_file = config.train_file, dict_file = config.dict_file, input_dict = False)
        self.test_data = reader(input_file = config.test_file, dict_file = config.dict_file, input_dict = True)
        self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'label')
        self.input_X = tf.placeholder(dtype = tf.int32, shape = [None, None])
        self.Build_model()


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
        self.train_data.load_data()
        self.test_data.load_data()
        train_X = np.asarray(self.train_data.words_index, dtype = np.int32)
        train_y = np.asarray(self.train_data.labels_index, dtype = np.int32)

        test_X = np.asarray(self.test_data.words_index, dtype = np.int32)
        test_y = np.asarray(self.test_data.labels_index, dtype = np.int32)

        train_dataSet = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        train_dataSet.shuffle(10000)
        train_dataSet = train_dataSet.batch(config.train_batch_size)

        test_dataSet = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        test_dataSet.shuffle(10000)
        test_dataSet = test_dataSet.batch(config.test_batch_size)

        self.Iterrator = tf.data.Iterator.from_structure(train_dataSet.output_types, train_dataSet.output_shapes)

        self.train_Initializer = self.Iterrator.make_initializer(train_dataSet)
        self.test_Initializer = self.Iterrator.make_initializer(test_dataSet)

    def Build_model(self):
        """
        构建分词模型
        输入层: input_X [batch_size, time_step]
               embedding [batch_size, time_step, embedding_size]
        BI-LSTM: 多层LSTM构成, 采用动态序列长度
                 input: [batch_size, input_size]
                 output: (outputs, output_state)
        """
        with tf.variable_scope("inputs"):
            # 求出batch_size中每个序列的长度, seq_length: [batch_size]
            self.seq_length = tf.reduce_sum(tf.sign(self.input_X), 1)
            self.seq_length = tf.cast(self.seq_length, tf.int32)
            embedding = tf.Variable(tf.random_normal([config.vocab_size, config.embedding_size]), dtype = tf.float32)
            weights_out = tf.Variable(tf.random_normal([2 * config.hidden_size, config.class_num]))
            bais_out = tf.Variable(tf.random_normal([config.class_num]))
            inputs = tf.nn.embedding_lookup(embedding, self.input_X)
        with tf.variable_scope("biLSTM"):
            cell_fw = rnn.MultiRNNCell([self.LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
            cell_bw = rnn.MultiRNNCell([self.LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
            bilstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs = inputs, sequence_length = self.seq_length, dtype = tf.float32)
            output = tf.reshape(tf.concat(bilstm_output, 2), [-1, config.hidden_size*2])
        logits = tf.matmul(output, weights_out) + bais_out
        # CRF模型
        self.scores = tf.reshape(logits, [-1, config.time_step, config.class_num])
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.input_y, self.seq_length)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        # 如果使用softmax则需要对loss做mask
        # losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(train_y, [-1]), logits = scores))
        # mask = tf.sequence_mask(seq_length)
        # losses = tf.boolean_mask(losses, mask
        # loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", self.loss)
        self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.loss, global_step = self.global_step)
        # temp_seq_length = seq_length
        # mask = tf.less(tf.expand_dims(tf.range(tf.reduce_max(temp_seq_length)), axis=0), tf.expand_dims(temp_seq_length, axis=1))
        # total_labels = tf.reduce_sum(temp_seq_length, axis=0)
        # correct_labels = tf.reduce_sum(tf.cast(tf.boolean_mask(tf.equal(input_y, viterbi_sequence), mask), tf.int32))
        # self.accuracy = tf.divide(correct_labels, total_labels)
        # tf.summary.scalar("accuracy", self.accuracy)


def train(model):
    """
    模型训练或者预测的主函数
    """

    model.load_data()
    train_num = model.train_data.sample_nums
    test_num = model.test_data.sample_nums
    model.input_X, model.input_y = model.Iterrator.get_next()
    saver = tf.train.Saver()
    with tf.Session(config = model.gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        for epoch in range(config.epochs):
            tf.train.global_step(sess, global_step_tensor = model.global_step)
            sess.run(model.train_Initializer)
            for train_batch in range(train_num // config.train_batch_size):
                correct_labels = 0
                total_labels = 0
                _, y, batch_loss, summary, gStep, seq_length, all_score, transition = sess.run([model.train_op, model.input_y, model.loss, merged, model.global_step, model.seq_len, model.scores, model.transition_params], feed_dict = {model.keep_prob : config.keep_prob})
                for (score, length, y_) in zip(all_score, seq_length, y):
                    y_ = y_[:length]
                    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(score[:length], transition)
                    correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                    total_labels += length
                accuracy = 100.0 * correct_labels / float(total_labels)
                if train_batch % config.per_summary == 0:
                    writer.add_summary(summary, epoch * (train_num // config.train_batch_size) + train_batch)
                if train_batch % config.per_print == 0:
                    print("Epoch: {} Global Step: {} training loss: {} batch_accuracy: {:.2f}%".format(epoch, gStep, batch_loss, accuracy))
            if epoch % config.per_test == 0:
                sess.run(model.test_Initializer)
                for test_batch in range(test_num // config.test_batch_size):
                    correct_labels = 0
                    total_labels = 0
                    _, y, seq_length, all_score, transition = sess.run([model.train_op, model.input_y, model.seq_len, model.scores, model.transition_params], feed_dict={model.keep_prob: config.keep_prob})
                    for (score, length, y_) in zip(all_score, seq_length, y):
                        y_ = y_[:length]
                        viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(score[:length], transition)
                        correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                        total_labels += length
                    accuracy = 100.0 * correct_labels / float(total_labels)
                    if test_batch % config.per_print == 0:
                        print("Test Accuracy: {:.2f}% step: {}".format(accuracy, test_batch))
            # 每 3 个 epoch 保存一次模型
            if epoch % config.per_save == 0:
                saver.save(sess, config.model_save_path, global_step = epoch)



def predict(model):
    """
    Introduction
    ------------
        加载训练好的模型进行预测
    Parameters
    ----------
        model: 模型结构
    Returns
    -------
        None
    """

    wordIndex = []
    ckpt = tf.train.get_checkpoint_state(config.model_ckpt)

    with open(config.predict_file, encoding='utf-8') as file:
        line = file.readlines()
        dataReader = reader(dict_file=config.dict_file, input_dict = True)
        for sentence in line:
            sentence = sentence.strip()
            tmp = dataReader.sentenceTowordIndex(sentence)
            for element in tmp:
                wordIndex.append(element)
    wordIndex = np.asarray(wordIndex, np.int32)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print(wordIndex)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load model: {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            x, seq_length, pred_scores, transition = sess.run([model.input_X, model.seq_length, model.scores, model.transition_params], feed_dict = {model.input_X : wordIndex})
            for (score, length) in zip(pred_scores, seq_length):
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(score[:length], transition)
                print(length, score, x)
                print(viterbi_sequence, viterbi_score)


def exportModelProtobuf(model):
    """
    Introduction
    ------------
        将模型和预测变量存储为pb文件
    Parameters
    ----------
        模型结构参数
    """
    builder = tf.saved_model.builder.SavedModelBuilder(config.export_dir)
    # 定义输入输出
    model_input = tf.saved_model.utils.build_tensor_info(model.input_X)
    scores = tf.saved_model.utils.build_tensor_info(model.scores)
    transition_params = tf.saved_model.utils.build_tensor_info(model.transition_params)
    seq_length = tf.saved_model.utils.build_tensor_info(model.seq_length)
    # 定义模型signature
    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'inputs': model_input},
        outputs={'scores': scores,
                 'transition_params' : transition_params,
                 'seq_length' : seq_length},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(config.model_ckpt)
    with tf.Session() as sess:
        if ckpt and ckpt.model_checkpoint_path:
            print("Load model: {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition})
            builder.save()
            print("Export model to {}".format(config.export_dir))

if __name__ == "__main__":
    model = BiLSTM_CRF()
    predict(model)
    exportModelProtobuf(model)
