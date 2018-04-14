"""
Introduction
------------
    模型训练参数配置文件
Author
------
    gaochen
Date
----
    2018.04.10
"""

# 模型参数设置
epochs = 10
batch_size = 64
# 语料库标注类别数
class_num = 4
# 隐藏藏层数
layer_num = 2
# 隐藏层神经元数
hidden_size = 256
# 句子padding最大长度
stence_len = 128
time_step = 128
# 语料库中字符数量
vocab_size = 4686
# embedding向量维度
embedding_size = 256
# Droupout保留概率
keep_prob = 0.5
# 学习率
lr = 0.0001
max_grad_norm = 5
# 句子字符最大程度
max_sentence_len = 128
# 输入数据集的文件路径
input_file = './data/pku_training.utf8'
# 词典输出文件
dict_file = './data/pku_training_dict.txt'
# 是否使用现有的词典
input_dict = False
# 模型保存路径
model_save_path = './model/'
GPU=0
