import re
import collections
from utils import strQ2B


class PreProcessing:
    def __init__(self, vocab_size, input_file, output_words_file, output_labels_file, dict_file, raw_file, input_dict=False):
        """
        构造函数
        Parameters
        ----------
            vocab_size: 语料中字符的数量
            input_file: 输入语料的文件路径
            output_words_file: 输入语料每个字在词汇表中对应的索引
            output_labels_file: 标签索引文件中的内容是输入语料中每个字对应的分词标签编号，采用SBIE标签，对应编号为0,1,2,3
            dict_file: 词典文件路径
            raw_file: 输出语料库为切分的原始文件路径
            input_dict: 指定是否输入词典，若为True，则使用dict_file指定的词典，若为False，则根据语料和vocab_size生成词典，并输出至dict_file指定的位置，默认为False
        """
        self.input_file = input_file
        self.output_words_file = output_words_file
        self.output_labels_file = output_labels_file
        self.dict_file = dict_file
        self.input_dict = input_dict
        self.vocab_size = vocab_size
        # 是否输出原始语料文件
        if raw_file is None or raw_file == '':
            self.output_raw_file = False
        else:
            self.output_raw_file = True
            self.raw_file = raw_file
        # 语料库中每个字出现的次数
        self.vocab_count = 0
        # 语料中每个字符数量，其中'UNK'表示词汇表外的字符，'STAT'表示句子首字符之前的字符，'END'表示句子尾字符后面的字符，这两字符用于生成字的上下文
        self.count = [['UNK', 0], ['STAT', 0],
                      ['END', 0]]
        # 字符分割符为双空格
        self.SPLIT_CHAR = '  '
        # 读取语料库中的句子
        self.sentences = self.read_sentence()
        self.words_index, self.labels_index = [], []
        # 根据是否指定词典路径来初始化词典，若指定，使用给定词典，未指定，根据语料生成
        # 词典中项表示字符编号，从0开始，{'UNK':0,'STAT':1,'END':2,'我':3,'们':4}
        if self.input_dict:
            self.dictionary = self.read_dictionary()
        else:
            self.dictionary = self.build_dictionary()


    def read_sentence(self):
        """
        读取文件中每行的句子，替换分割符
        """
        with open(self.input_file, encoding = 'utf-8') as file:
            line = file.read()
            # 将所有中文标点符号转换为英文标点符号
            sentences = strQ2B(line).splitlines()
            sentences = list(filter(None, sentences))
        return sentences


    def build_raw_corpus(self):
        """
        保存预处理之后的语料库
        """
        with open(self.raw_file, 'w', encoding = 'utf-8') as file:
            for sentence in self.sentences:
                file.write(sentence.replace(' ', '') + '\n')


    def build_dictionary(self):
        """
        建立词汇表字典, 首先统计每个字出现的次数，然后取出出现次数最大的前VOCAB_SIZE个字
        """
        dictionary = {}
        words = ''.join(self.sentences).replace(' ', '')
        self.vocab_count = len(collections.Counter(words))
        self.count.extend(collections.Counter(words).most_common(self.vocab_size - 3))
        for word, _ in self.count:
            dictionary[word] = len(dictionary)
        return dictionary


    def read_dictionary(self):
        """
        读取词汇表字典
        """
        with open(self.dict_file, 'r', encoding = 'utf-8') as file:
            dict_content = file.read().splitlines()
            dictionary = {}
            dict_arr = [item.split() for item in dict_content]
            for word in dict_arr:
                dictionary[word[0]] = word[1]
        return dictionary


    def build_basic_dataset(self):
        """

        :return:
        """
        unk_count = 0
        # 语料库中句子进行标号
        for sentence in self.sentences:
            sentence = sentence.replace(' ', '')
            sen_data = []
            for word in sentence:
                if word in self.dictionary:
                    index = self.dictionary[word]
                else:
                    index = 0
                    unk_count += 1
                sen_data.append(index)
            self.words_index.append(sen_data)
        self.count[0][1] = unk_count

    def build_corpus_dataset(self):
        """

        """
        empty = 0
        for sentence in self.sentences:
            sentence_label = []
            words = sentence.strip().split(self.SPLIT_CHAR)
            for word in words:
                l = len(word)
                if l == 0 :
                    empty += 1
                    continue
                elif l == 1:
                    sentence_label.append(0)
                else:
                    sentence_label.append(l)
                    sentence_label.extend([2] * (l - 2))
                    sentence_label.append(3)
            self.labels_index.append(sentence_label)

    def buid_test_corpus(self, filename):
        """

        :param filename:
        :return:
        """
        with open(filename, 'w', encoding = 'utf-8') as file:
            for (sentence, sentence_label) in enumerate(zip(self.sentences, self.labels_index)):
                file.write(sentence.replace(' ', '') + '\n')
                file.write(' '.join([str(item) for item in sentence_label]) + '\n')

    def build_exec(self):
        """

        :return:
        """
        self.build_basic_dataset()
        self.build_corpus_dataset()
        with open(self.output_words_file, 'w+', encoding = 'utf-8') as words_file:
            with open(self.output_labels_file, 'w+', encoding = 'utf-8') as labels_file:
                for (words, labels) in enumerate(zip(self.words_index, self.labels_index)):
                    words_file.write(' '.join(str(word) for word in words) + '\n')
                    labels_file.write(' '.join(str(label) for label in labels) + '\n')
                if not self.input_dict:
                    dict_file = open(self.dict_file, 'w+', encoding = 'utf-8')
                    for (word, index) in self.dictionary.items():
                        dict_file.write(word + ' ' + str(index) + '\n')
                    dict_file.close()
        if self.output_raw_file:
            self.build_raw_corpus()

if __name__ == "__main__":
    prepare_pku = PreProcessing(4000, './corpus/pku_training.utf8', './corpus/pku_training_words.txt',
                            './corpus/pku_training_labels.txt', './corpus/pku_training_dict.txt', 'corpus/pku_training_raw.utf8')
    prepare_pku.build_exec()