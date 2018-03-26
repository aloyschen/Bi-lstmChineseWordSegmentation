# 该脚本是用lstm实现回归预测
# Author: gaochen
# data: 2018.03.09

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BATCH_START = 0

def get_batch_data(batch_size, time_steps):
    """
    获取训练集数据
    Parameters
    ----------

    """
    global BATCH_START
    x = np.arange(BATCH_START, BATCH_START + batch_size * time_steps).reshape([batch_size, time_steps]) / (10 * np.pi)
    seq = np.sin(x)
    # res = np.sin(x + 1)
    res = np.cos(x)
    BATCH_START += time_steps
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], x]


class lstm_model:
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr = 0.1):
        """
        构建lstm模型
        Parameters
        ----------
            n_steps: 每个样本的时间序列长度
            input_size: 每个时间点对应输入的维度
            output_size: 和输入维度相对应
            batch_size: 每批训练样本数目
            lr: 学习率
        Returns
        -------
            None
        """
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.lr = lr
        self.cell_size = cell_size


    def add_input_layer(self, xs):
        """
        定义输入层结构，后续连接LSTM隐藏层
        :return:
        """
        #首先需要将输入的参数维度转为二维，即【batch_size, time_steps, input_size】-> 【batch_size*time_steps, input_size】
        input_x = tf.reshape(xs, [-1, self.input_size])
        weight_input = self.get_weights([self.input_size, self.cell_size])
        bias_input = self.get_bais(self.cell_size)
        with tf.name_scope("wx_plus_b"):
            input_y = tf.matmul(input_x, weight_input) + bias_input
        #再将输出层转换为之前的维度
        input_y = tf.reshape(input_y, [-1, self.n_steps, self.cell_size])
        return input_x, input_y


    def add_LSTM_layer(self, input_y):
        """
        tensorflow的lstm每个batchsize会保存一个最终的finalState，作为下一个batchsize的初始值
        """
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, state_is_tuple = True)
        #定义最初始的状态值
        with tf.name_scope('intial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype = tf.float32)
        cell_outputs, cell_final_state = tf.nn.dynamic_rnn(lstm_cell, input_y,
                            initial_state = self.cell_init_state, time_major = False)
        print("input_y:", input_y.shape)
        print("cell_outputs.shape:", cell_outputs.shape)
        return cell_outputs, cell_final_state

    def add_output_layer(self, cell_outputs):
        """
        定义输出层的网络结构
        """
        output_x = tf.reshape(cell_outputs, [-1, self.cell_size])
        weight_out = self.get_weights([self.cell_size, self.output_size])
        bias_out = self.get_bais(self.output_size)
        with tf.name_scope("wx_plus_b"):
            output_y = tf.matmul(output_x, weight_out) + bias_out
        return output_y


    def compute_loss(self, output_y, input_y):
        """
        计算模型损失函数
        """
        losses = tf.square(tf.subtract(tf.reshape(tf.tanh(output_y), [-1]), tf.reshape(input_y, [-1])))
        #因为求出的losses是所有的cell，要求平均值
        with tf.name_scope('averange_loss'):
            loss = tf.div(tf.reduce_sum(losses), self.batch_size)
            tf.summary.scalar('loss', loss)
        return loss



    def get_weights(self, shape, w_name = "weights"):
        """
        初始化权重矩阵
        Parameters
        ----------
            shape: 矩阵大小
            name: 名字
        """
        initiatizer = tf.random_normal_initializer(mean = 0., stddev = 1.)
        return tf.get_variable(shape = shape, initializer = initiatizer, name = w_name)


    def get_bais(self, shape, b_name="bias"):
        """
        初始化偏置矩阵
        Parameters
        ----------
            shape: 矩阵大小
            name: 名字
        """
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape = shape, initializer = initializer, name = b_name)





    def train(self, Logdir, iter_num):
        """
        定义整个模型结构，并训练LSTM模型
        """
        with tf.name_scope("inputs"):
            xs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name = 'xs')
            ys = tf.placeholder(tf.float32, [None, self.n_steps, self.output_size], name = 'ys')
        with tf.variable_scope("hidden"):
            input_x, input_y = self.add_input_layer(xs)
        with tf.variable_scope("LSTM_Cell"):
            cell_outputs, cell_final_state = self.add_LSTM_layer(input_y)
        with tf.variable_scope("output_hidden"):
            output_y = self.add_output_layer(cell_outputs)
        with tf.variable_scope("loss"):
            loss = self.compute_loss(output_y, ys)
        with tf.variable_scope("train"):
            train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(loss)
        sess = tf.Session()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(Logdir, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        for iter in range(iter_num):
            seq, res, x_range = get_batch_data(self.batch_size, self.n_steps)
            if iter == 0:
                feed_dict = {ys : res, xs : seq}
            else:
                feed_dict = {ys : res, xs : seq, self.cell_init_state : state}
            _, cost, state, pred = sess.run([train_op, loss, cell_final_state, output_y], feed_dict = feed_dict)
            # print("x:{}   y:{}   pred:{}".format(x_range[0, :], res[0].flatten(), pred.flatten()[:self.n_steps]))
            # print(x_range.shape, '\n', pred.flatten().shape)

            if iter % 200 == 0:
                plt.plot(x_range[0, :], res[0].flatten(), 'r', x_range[0, :], pred.flatten()[:self.n_steps], 'b--')
                plt.ylim((-1.2, 1.2))
                plt.draw()
                plt.pause(0.3)
                plt.show()
                print('loss:', round(cost, 4))
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, iter)





