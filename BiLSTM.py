"""
Introduction: 构建BiLSTM + CRF 模型进行中文分词
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
    """
    定义每一层LSTM单元结构，每一层中加入一层dropout
    """
    cell = rnn.BasicLSTMCell(config.hidden_size, reuse=tf.AUTO_REUSE)
    dropout = rnn.DropoutWrapper(cell, config.keep_prob)
    return dropout


def BiLSTM(train_X, train_y):
    """
    构建分词模型
    输入层: train_X [batch_size, time_step]
           embedding [batch_size, time_step, embedding_size]
    BI-LSTM: 多层LSTM构成, 采用动态序列长度
             input: [batch_size, input_size]
             output: (outputs, output_state)
    """
    with tf.variable_scope("inputs"):
        # 求出batch_size中每个序列的长度, seq_length: [batch_size]
        seq_length = tf.reduce_sum(tf.sign(train_X), 1)
        seq_length = tf.cast(seq_length, tf.int32)
        embedding = tf.Variable(tf.random_normal([config.vocab_size, config.embedding_size]), dtype = tf.float32)
        weights_out = tf.Variable(tf.random_normal([2 * config.hidden_size, config.class_num]))
        bais_out = tf.Variable(tf.random_normal([config.class_num]))
        inputs = tf.nn.embedding_lookup(embedding, train_X)
    with tf.variable_scope("biLSTM"):
        cell_fw = rnn.MultiRNNCell([LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([LSTM_Cell() for _ in range(config.layer_num)], state_is_tuple=True)
        bilstm_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs = inputs, sequence_length = seq_length, dtype = tf.float32)
        output = tf.reshape(tf.concat(bilstm_output, 2), [-1, config.hidden_size*2])
    logits = tf.matmul(output, weights_out) + bais_out
    # CRF模型
    scores = tf.reshape(logits, [-1, config.max_sentence_len, config.class_num])
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, train_y, seq_length)
    loss = tf.reduce_mean(-log_likelihood)
    # 如果使用softmax则需要对loss做mask
    # losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(train_y, [-1]), logits = scores))
    # mask = tf.sequence_mask(seq_length)
    # losses = tf.boolean_mask(losses, mask
    # loss = tf.reduce_mean(losses)
    tf.summary.scalar("loss", loss)
    train_op = tf.train.AdamOptimizer(config.lr).minimize(loss)
    correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(train_y, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return train_op, accuracy, loss


def main():
    """
    模型训练或者预测的主函数
    """
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    input_x = tf.placeholder(tf.int32, [None, config.max_sentence_len], name = "input_X")
    input_y = tf.placeholder(tf.int32, [None, config.max_sentence_len], name = "input_y")
    train_op, accuracy, loss = BiLSTM(input_x, input_y)
    saver = tf.train.Saver()
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        if config.train == True:
            data = reader(config.train_file, config.dict_file, False)
            for epoch in range(config.epochs):
                for batch in range(data.sample_nums // config.batch_size):
                    batch_X, batch_y = data.get_batch(config.batch_size)
                    feed_dict = {input_x : batch_X, input_y : batch_y}
                    _, batch_accuracy, batch_loss  = sess.run([train_op, accuracy, loss], feed_dict = feed_dict)
                    if batch % 100 == 0:
                        print("Epoch: {} iter_num: {} training loss: {} batch_accuracy: {}".format(epoch, batch, batch_loss, batch_accuracy))
                # 每 3 个 epoch 保存一次模型
                if (epoch + 1) % 3 == 0:
                    saver.save(sess, config.model_save_path, global_step = epoch)
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
    main()
