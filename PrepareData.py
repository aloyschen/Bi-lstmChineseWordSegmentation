import re
import collections
import config
import numpy as np
from utils import strQ2B

class reader:
    def __init__(self, input_file, dict_file, input_dict=False):
        """
        构造函数
        Parameters
        ----------
            vocab_size: 语料中字符的数量
            input_file: 输入语料的文件路径, 可能是训练集或者测试集
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
        # 记录batch的index
        self.index_in_epoch = 0
        # 完成了多少次epochs
        self.epochs_completed = 0
        # 样本数据量
        self.sample_nums = 0
        # 数据集数据处理
        self.read_words()
        self.build_vocab()
        self.word_to_ids()
        self.word_label()


    def read_words(self):
        """
        读取语料库中每个句子的字符
        """
        with open(self.input_file, 'r', encoding = 'utf-8') as file:
            # 将所有中文标点符号转换为英文标点符号
            data = strQ2B(file.read())
            self.sentences = data.splitlines()
            # 根据标点符号对长句子进行切分
            self.sentences = re.split(u'[。，？；！]', ''.join(self.sentences))
            self.sentences = [sentence.strip() for sentence in self.sentences if len(sentence) >= 3]
            self.sample_nums = len(self.sentences)
            words = data.replace('\n', "").split(self.SPLIT_CHAR)
            words = [word.strip() for word in words]
            self.words = [char for word in words for char in word ]



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
            self.dictionary = dict(zip(words, range(1, len(words))))
            self.save_dictionary()


    def word_to_ids(self):
        """
        获取所有字符的对应index列表
            1、需要进行padding处理，如果句子字符长度超过max则舍弃
            2、如果句子字符长度小于max则补0
        """
        for sentence in self.sentences:
            tmp = []
            sentence = sentence.replace(self.SPLIT_CHAR, '')
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
                    sentence_label.append(1)
                else:
                    sentence_label.append(2)
                    sentence_label.extend([3] * (l - 2))
                    sentence_label.append(4)
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

    def index_str(self, word_index, labels=None):
        """
        将输入的每个字符的index转换为对应的汉字
        把对应的label也转换为"BMSE"
        Parameters
        ----------
            word_index: 每个字符的index
            label: 每个字符对应的标签
        """
        label_dict = {1 : 'S', 2 : 'B', 3 : 'M', 4 : 'E'}
        word_str = []
        label_str = []
        # 将padding加入的0元素删除
        word_index = [element for element in word_index if element != 0]
        for word in word_index:
            if word in self.dictionary.keys():
                word_str.append(list(self.dictionary.keys())[list(self.dictionary.values()).index(word)])
            else:
                print("Don't have this word")
        if labels != None:
            labels = [element for element in labels if element != 0]
            for label in labels:
                label_str.append(label_dict[label])
            result = [word_str_per + '/' + label_str_per for word_str_per, label_str_per in zip(word_str, label_str)]
            print(' '.join(result))
        else:
            print(' '.join(word_str))


    def get_batch(self, batch_size):
        """
        产生tensorflow训练和预测所需的数据
        Parameters
        ----------
            batch_size: 训练或者预测时batch的大小
        """
        start = self.index_in_epoch
        num_samples = len(self.words_index)
        self.words_index = np.asarray(self.words_index)
        self.labels_index = np.asarray(self.labels_index)
        if start == 0 and self.epochs_completed == 0:
            idx = np.arange(0, num_samples)
            np.random.shuffle(idx)
            self.words_index = self.words_index[idx]
            self.labels_index = self.labels_index[idx]
        if start + batch_size > num_samples:
            self.epochs_completed += 1
            # 如果最后一个batch_size不够，还剩下一些数据
            rest_num_samples = num_samples - start
            rest_words_index = self.words_index[start:rest_num_samples]
            rest_label_index = self.labels_index[start:rest_num_samples]
            idx_new = np.arange(0, num_samples)
            np.random.shuffle(idx_new)
            self.words_index = self.words_index[idx_new]
            self.labels_index = self.labels_index[idx_new]
            start = 0
            self.index_in_epoch = batch_size - rest_num_samples
            end = self.index_in_epoch
            batch_word_index =  np.concatenate((self.words_index[start:end], rest_words_index), axis = 0)
            batch_label_index = np.concatenate((self.labels_index[start:end], rest_label_index), axis = 0)
            return batch_word_index, batch_label_index

        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.words_index[start:end], self.labels_index[start:end]
if __name__ == '__main__':
    data = reader(config.train_file, config.dict_file, False)
    print(data.sentences[0])
    print(data.words_index[0], len(data.words_index[0]))
    print(data.labels_index[0], len(data.labels_index[0]))
