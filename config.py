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
epochs = 100
batch_size = 64
# 语料库标注类别数
class_num = 5
# 隐藏藏层数
layer_num = 2
# 隐藏层神经元数
hidden_size = 64
# 句子padding最大长度
time_step = 32
# 语料库中字符数量
# vocab_size = 4686
vocab_size = 5152
# embedding向量维度
embedding_size = 64
# Droupout保留概率
keep_prob = 0.5
# 学习率
lr = 0.001
max_grad_norm = 5
# 句子字符最大程度
max_sentence_len = 32
# 训练数据集的文件路径
train_file = './data/pku_training.utf8'
# 测试数据集的文件路径
test_file = './data/pku_test.utf8'
# 词典输出文件
dict_file = './data/pku_training_dict.txt'
# 模型保存路径
model_save_path = './model/BiLSTM.ckpt'
# 模型的ckpt文件路径
model_ckpt = './model/'
# 使用GPU的型号
GPU=0
# 使用模型训练或者预测
train=True
