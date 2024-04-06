import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LSTMForNER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(LSTMForNER, self).__init__()
        self.hidden_dim = hidden_dim # 隐藏层维度
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True) # lstm层
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)#  隐藏层

    def forward(self, sentence):  # 这个函数是一个前向传播函数，用于计算给定句子的标签概率分数
        embeds = self.word_embeddings(sentence)  #使用self.word_embeddings将句子中的单词转换为嵌入向量。
        lstm_out, _ = self.lstm(embeds) # 使用self.lstm将嵌入向量转换为LSTM输出
        tag_space = self.hidden2tag(lstm_out) # 使用self.hidden2tag将LSTM输出转换为标签空间
        tag_scores = torch.log_softmax(tag_space, dim=-1) # 使用torch.log_softmax将标签空间转换为标签概率分数
        return tag_scores
