'''
@Author: xiaoyao jiang
@Date: 2020-04-08 15:17:27
@LastEditTime: 2020-07-06 15:24:06
@LastEditors: xiaoyao jiang
@Description: all model config
@FilePath: /bookClassification/src/utils/config.py
'''
import torch
import os
import numpy as np

# generate config
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]

train_file = root_path + '/data/train_clean.tsv'
dev_file = root_path + '/data/dev_clean.tsv'
test_file = root_path + '/data/test_clean.tsv'
stopWords_file = root_path + '/data/stopwords.txt'
log_dir = root_path + '/logs/'

# generate dl config
embedding = 'random'
embedding_pretrained = torch.tensor(
                       np.load(root_path + '/data/' + embedding)["embeddings"].astype('float32')) \
                       if embedding != 'random' else None

is_cuda = True
device = torch.device('cuda') if is_cuda else torch.device('cpu')
class_list = [
    x.strip() for x in open(root_path + '/data/class.txt').readlines()
]  # 类别名单
num_classes = len(class_list)

num_epochs = 30  # epoch数
batch_size = 32  # mini-batch大小
pad_size = 400  # 每句话处理成的长度(短填长切)
learning_rate = 2e-5  # 学习率
dropout = 1.0  # 随机失活
require_improvement = 10000  # 若超过1000batch效果还没提升，则提前结束训练
n_vocab = 50000  # 词表大小，在运行时赋值
embed = 300  # 向量维度
hidden_size = 512  # lstm隐藏层
num_layers = 1  # lstm层数
eps = 1e-8
max_length = 400
dim_model = 300
hidden = 1024
last_hidden = 512
num_head = 5
num_encoder = 2

# explain ai
model_type = 'bert'
max_seq_length = 250
do_lower_case = True
per_gpu_train_batch_size = 8
per_gpu_eval_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 5e-5
weight_decay = 1.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = 5.0
max_steps = -1
warmup_steps = 0
start_pos = 0
end_pos = 2000
visualize = -1
seed = 111