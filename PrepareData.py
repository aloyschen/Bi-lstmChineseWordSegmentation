import re
import collections
import config
from utils import strQ2B
import tensorflow as tf


class reader:
    def __init__(self, input_file, dict_file, input_dict=False):
        """
        构造函数
        Parameters
        ----------
            vocab_size: 语料中字符的数量
            input_file: 输入语料的文件路径, 可能是训练集或者测试集
            output_words_file: 输入语料每个字在词汇表中对应的索引
            output_labels_file: 标签索引文件中的内容是输入语料中每个字对应的分词标签编号，采用SBIE标签，对应编号为0,1,2,3
            dict_file: 词典文件存储路径
            input_dict: 指定是否输入词典，若为True，则使用dict_file指定的词典，若为False，则根据语料和vocab_size生成词典，并输出至dict_file指定的位置，默认为False
        """
        self.input_file = input_file
        self.dict_file = dict_file
        self.input_dict = input_dict
        # 存储语料库中的每个字符
        self.words = []
        # 分割符为双空格
        self.SPLIT_CHAR = '  '
        # 语料库中的句子
        self.sentences = []
        # word_index存储了每句话中每个字符在字典中的序号
        self.words_index, self.labels_index = [], []


    def read_words(self):
        """
        读取语料库中每个句子的字符
        """
        with open(self.input_file, 'r', encoding = 'utf-8') as file:
            # 将所有中文标点符号转换为英文标点符号
            data = strQ2B(file.read())
            self.sentences = data.splitlines()
            # 去除空行
            self.sentences = list(filter(None, self.sentences))
            words = data.replace('\n', "").split(self.SPLIT_CHAR)
            self.words = [char for word in words for char in word]



    def build_vocab(self):
        """
        建立词汇表
            1、首先统计每个字出现的次数
            2、取出出现次数最大的前VOCAB_SIZE个字符(出现次数排序后)
            3、获取每个字符对应的index
        """
        # 根据是否指定词典路径来初始化词典，若指定，使用给定词典，未指定，根据语料生成
        # 词典中项表示字符编号，从0开始，{'我':3,'们':4,'不':5}
        if self.input_dict:
            self.dictionary = self.read_dictionary()
        else:
            counter = collections.Counter(self.words)
            count_pairs = sorted(counter.items(), key = lambda x : -x[1])
            words, _ = list(zip(*count_pairs))
            self.dictionary = dict(zip(words, range(len(words))))
            self.save_dictionary()

    def word_to_ids(self):
        """
        获取所有字符的对应index列表
            1、需要进行padding处理，如果句子字符长度超过max则舍弃
            2、如果句子字符长度小于max则补0
        """
        for sentence in self.sentences:
            tmp = []
            sentence.replace(self.SPLIT_CHAR, '')
            for word in sentence:
                if word in self.dictionary:
                    tmp.append(self.dictionary[word])
            if len(tmp) >= config.max_sentence_len:
                self.words_index.append(tmp[:config.max_sentence_len])
            else:
                tmp.extend([0] * (config.max_sentence_len - len(tmp)))
                self.words_index.append(tmp)


    def word_label(self):
        """
        根据语料中每个词的长度取每个字符对应的label
            1、如果是单个字符对应的标签为0
            2、如果词的长度大于2则根据BME标注，B对应的标签为1, M对应的标签为2，E对应的标签为3
            3、对标签同样做padding处理
        """
        for sentence in self.sentences:
            sentence_label = []
            for word in sentence.split(self.SPLIT_CHAR):
                l = len(word)
                if l == 0:
                    continue
                elif l == 1:
                    sentence_label.append(0)
                else:
                    sentence_label.append(1)
                    sentence_label.extend([2] * (l - 2))
                    sentence_label.append(3)
            if len(sentence_label) >= config.max_sentence_len:
                self.labels_index.append(sentence_label[:config.max_sentence_len])
            else:
                sentence_label.extend([0] * (config.max_sentence_len - len(sentence_label)))
                self.labels_index.append(sentence_label)



    def save_dictionary(self):
        """
        存储构建的词典
        """
        with open(self.dict_file, 'w+', encoding = 'utf-8') as file:
            for word in self.dictionary.keys():
                file.write(str(word) + '\t' + str(self.dictionary[word]) + '\n')

    def read_dictionary(self):
        """
        读取指定的词汇表字典
        """
        with open(self.dict_file, 'r', encoding = 'utf-8') as file:
            dict_content = file.read().splitlines()
            dictionary = {}
            dict_arr = [item.split('\t') for item in dict_content]
            for word in dict_arr:
                dictionary[word[0]] = word[1]
        return dictionary

    def get_batch(self, batch_size, num_steps):
        """
        产生tensorflow所需的数据
        """
        self.read_words()
        self.build_vocab()
        self.word_to_ids()
        self.word_label()

        with tf.name_scope("data_producer"):
            print(len(self.words_index), len(self.labels_index))
            dataset = tf.data.Dataset.from_sparse_tensor_slices((self.words_index, self.labels_index))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print(sess.run(dataset))


if __name__ == "__main__":
    prepare_pku = reader('./data/pku_training.utf8', './data/pku_training_dict.txt')
    prepare_pku.get_batch(batch_size = 128, num_steps = 10)