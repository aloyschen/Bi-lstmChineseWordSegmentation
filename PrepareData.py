import re
import collections
from utils import strQ2B


class PreProcessing:
    def __init__(self, vocab_size, input_file, output_words_file, output_labels_file, dict_file, raw_file, input_dict=False):
        """
        构造函数
        Parameters
        ----------
            vocab_size: 词汇表大小
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
            output_raw_file = False
        else:
            output_raw_file = True
            self.raw_file = raw_file
        # 语料库中字符数量
        self.vocab_count = 0
        # 字符分割符为双空格
        self.SPLIT_CHAR = '  '
        # 读取语料库中的句子
        self.sentence = self.read_sentence()
        self.words_index, self.labels_index = [], []


    def read_sentence(self):
        """

        :return:
        """
        with open(self.input_file, encoding = 'utf-8') as file:
            line = file.read()
            # 将分割符统一成双空格
            sentences = re.sub("[ ]+", self.SPLIT_CHAR, strQ2B(line)).splitlines()
            sentences = list(filter(None, sentences))
        return sentences
