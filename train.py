import config
import os
from BiLSTMCRF import BiLSTM_CRF
from PrepareData import reader
import numpy as np
import tensorflow as tf

# gpu的配置
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


def load_data():
    """
    Introduction
    ------------
        加载数据, 返回训练集和测试集数据的迭代器
    Parameters
    ----------
        train_file: 训练集路径
        test_file: 测试集路径
        dict_file: 使用字典路径
        input_dict: 是否使用输入字典
    Returns
    -------
        train_Initializer: 训练集数据初始
        test_Initializer: 测试集数据
        Iterator: 数据集迭代器
    """
    train_data = reader(input_file = config.train_file, dict_file = config.dict_file, input_dict = config.input_dict)
    test_data = reader(input_file = config.test_file, dict_file = config.dict_file, input_dict = config.input_dict)
    train_data.load_data()
    test_data.load_data()
    train_X = np.asarray(train_data.words_index, dtype=np.int32)
    train_y = np.asarray(train_data.labels_index, dtype=np.int32)

    test_X = np.asarray(test_data.words_index, dtype=np.int32)
    test_y = np.asarray(test_data.labels_index, dtype=np.int32)

    train_dataSet = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataSet.shuffle(10000)
    train_dataSet = train_dataSet.batch(config.train_batch_size)

    test_dataSet = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    test_dataSet.shuffle(10000)
    test_dataSet = test_dataSet.batch(config.test_batch_size)

    Iterator = tf.data.Iterator.from_structure(train_dataSet.output_types, train_dataSet.output_shapes)

    train_Initializer = Iterator.make_initializer(train_dataSet)
    test_Initializer = Iterator.make_initializer(test_dataSet)
    return train_Initializer, test_Initializer, Iterator, train_data, test_data

def train():
    """
    Introduction
    ------------
        该函数用于训练BILSTM—CRF模型
    Parameters
    ----------
        None
    Returns
    -------
        None
    """
    model = BiLSTM_CRF()

    train_Initializer, test_Initializer, Iterator, train_data, test_data = load_data()
    train_num = train_data.sample_nums
    test_num = test_data.sample_nums
    input_X, input_y = Iterator.get_next()
    model.Build_model(input_X, input_y, config.vocab_size, config.embedding_size, config.hidden_size, config.class_num, config.layer_num, config.time_step, config.lr)
    saver = tf.train.Saver()
    gpu_config = tf.ConfigProto()
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        for epoch in range(config.epochs):
            tf.train.global_step(sess, global_step_tensor=model.global_step)
            sess.run(train_Initializer)
            for train_batch in range(train_num // config.train_batch_size):
                correct_labels = 0
                total_labels = 0
                _, y, batch_loss, summary, gStep, seq_length, all_score, transition = sess.run([model.train_op, input_y, model.loss, merged, model.global_step, model.seq_length, model.scores, model.transition_params], feed_dict={model.keep_prob: config.keep_prob})
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
                sess.run(test_Initializer)
                for test_batch in range(test_num // config.test_batch_size):
                    correct_labels = 0
                    total_labels = 0
                    _, y, seq_length, all_score, transition = sess.run(
                        [model.train_op, input_y, model.seq_length, model.scores, model.transition_params],
                        feed_dict={model.keep_prob: config.keep_prob})
                    for (score, length, y_) in zip(all_score, seq_length, y):
                        y_ = y_[:length]
                        viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(score[:length], transition)
                        correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                        total_labels += length
                    accuracy = 100.0 * correct_labels / float(total_labels)
                    if test_batch % config.per_print == 0:
                        print("Test Accuracy: {:.2f}% step: {}".format(accuracy, test_batch))
            # 每 3 个 epoch 保存一次模型
            # if epoch % config.per_save == 0:
            #     saver.save(sess, config.model_save_path, global_step=epoch)


if __name__ == '__main__':
    train()