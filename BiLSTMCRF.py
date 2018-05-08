"""
Introduction: 构建模型BiLSTM + CRF
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
        # self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'label')
        # self.input_X = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'word')


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


    def Build_model(self, input_X, input_y, vocab_size, embedding_size, hidden_size, class_num, layer_num, time_step, lr):
        """
        Introduction
        ------------
            构建分词模型
            输入层: input_X [batch_size, time_step]
                   embedding [batch_size, time_step, embedding_size]
            BI-LSTM: 多层LSTM构成, 采用动态序列长度
                     input: [batch_size, input_size]
                     output: (outputs, output_state)
        Parameters
        ----------
            vocab_size: 词汇表的大小
            embedding_size: embedding的大小
            hidden_size: 隐藏层单元数量
            class_num: 类别数目
            layer_num: BILSTM层数
            time_step: 时间维度大小, 即每句话长度
            lr: 学习率
        """
        with tf.variable_scope("inputs"):
            # 求出batch_size中每个序列的长度, seq_length: [batch_size]
            self.seq_length = tf.reduce_sum(tf.sign(input_X), 1)
            self.seq_length = tf.cast(self.seq_length, tf.int32)
            embedding = tf.Variable(tf.random_normal([vocab_size, embedding_size]), dtype = tf.float32)
            weights_out = tf.Variable(tf.random_normal([2 * hidden_size, class_num]))
            bais_out = tf.Variable(tf.random_normal([class_num]))
            inputs = tf.nn.embedding_lookup(embedding, input_X)
        with tf.variable_scope("biLSTM"):
            cell_fw = rnn.MultiRNNCell([self.LSTM_Cell() for _ in range(layer_num)], state_is_tuple=True)
            cell_bw = rnn.MultiRNNCell([self.LSTM_Cell() for _ in range(layer_num)], state_is_tuple=True)
            bilstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs = inputs, sequence_length = self.seq_length, dtype = tf.float32)
            output = tf.reshape(tf.concat(bilstm_output, 2), [-1, hidden_size*2])
        logits = tf.matmul(output, weights_out) + bais_out
        # CRF模型
        self.scores = tf.reshape(logits, [-1, time_step, class_num])
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, input_y, self.seq_length)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        tf.summary.scalar("loss", self.loss)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step = self.global_step)






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
