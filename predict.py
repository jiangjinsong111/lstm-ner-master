import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datautils import NERDataset
from models import LSTMForNER
from datautils import train_data,word_to_ix,test_data,tag_to_ix,collate_fn,ix_to_tag
from sklearn.metrics import precision_recall_fscore_support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义文本预处理函数
def preprocess_sentence(sentence, word_to_ix, max_length):
    # 分词 (这里假设输入的句子已经是分词后的列表形式)
    indexed_sentence = [word_to_ix.get(word, word_to_ix["<UNK>"]) for word in sentence]
    # 填充或截断
    if len(indexed_sentence) < max_length:
        indexed_sentence += [word_to_ix["<PAD>"]] * (max_length - len(indexed_sentence))
    else:
        indexed_sentence = indexed_sentence[:max_length]
    return torch.tensor([indexed_sentence], dtype=torch.long)


model = LSTMForNER(len(word_to_ix), len(tag_to_ix), embedding_dim=64, hidden_dim=128)
model.to(device)
# 获取用户输入
raw_text = input("请输入文本：")

# 使用jieba进行分词
sentence = list(jieba.cut(raw_text))

# 接下来是预处理和模型预测的代码
# 注意：下面的代码仅为示例，确保它与你的模型和预处理流程兼容

# 假设word_to_ix, tag_to_ix, ix_to_tag, model等都已经准备好
max_length = 10  # 假设最大长度是10，这应该与训练时相匹配

# 预处理句子
preprocessed_sentence = preprocess_sentence(sentence, word_to_ix, max_length).to(device)

# 确保模型在评估模式
model.eval()

with torch.no_grad():  # 不计算梯度
    output = model(preprocessed_sentence)
    predictions = torch.argmax(output, dim=2)  # 获取最可能的标签索引

# 转换预测的索引回标签
predicted_tags = [ix_to_tag[ix] for ix in predictions[0].cpu().numpy()]  # 假设我们处理的是第一个句子

print("原始句子:", raw_text)
print("分词结果:", "/ ".join(sentence))
print("预测标签:", "/ ".join(predicted_tags))
