

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader



import torch
from torch.utils.data import Dataset, DataLoader

class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = [self.word_to_ix[word] if word in self.word_to_ix else self.word_to_ix["<UNK>"] for word in self.sentences[idx]]
        tag = [self.tag_to_ix[tag] for tag in self.tags[idx]]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(tag, dtype=torch.long)

    @staticmethod
    def read_data(file_path):
        sentences, labels = [], []
        with open(file_path, encoding='utf-8') as f:
            sentence, label = [], []
            for line_num, line in enumerate(f, 1):  # 添加行号以帮助定位问题
                line = line.strip()
                if not line:
                    if sentence and label:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence, label = [], []
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        word, tag = parts
                        sentence.append(word)
                        label.append(tag)
                    # else:
                        # print(f"Format error in line {line_num}: {line}")  # 打印出错的行
            if sentence and label:  # 添加最后一个句子，如果存在
                sentences.append(sentence)
                labels.append(label)
        return sentences, labels

    @staticmethod
    def build_vocab(tags, sentences):
        tag_to_ix = {"<PAD>": 0}  # 初始化标签字典
        word_to_ix = {"<PAD>": 0, "<UNK>": 1}  # 初始化单词字典
        for tag in tags:
            for label in tag:
                if label not in tag_to_ix:
                    tag_to_ix[label] = len(tag_to_ix)
        for sentence in sentences:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        return word_to_ix, tag_to_ix

# 定义 NERDataset 类（代码省略，使用上面定义的类）

# 首先，读取并预处理训练数据和测试数据
sentences_train, tags_train = NERDataset.read_data('./data/train.txt')
sentences_test, tags_test = NERDataset.read_data('./data/test.txt')

# 构建词汇表，通常基于训练数据构建，确保训练和测试使用相同的词汇表
word_to_ix, tag_to_ix = NERDataset.build_vocab(tags_train, sentences_train)

# 创建训练数据集和测试数据集实例
train_data = NERDataset(sentences_train, tags_train, word_to_ix, tag_to_ix)
test_data = NERDataset(sentences_test, tags_test, word_to_ix, tag_to_ix)


# 自定义 collate_fn 来处理数据批次
def collate_fn(batch):
    sentences, tags = zip(*batch)
    # 填充句子
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_to_ix["<PAD>"])
    # 填充标签
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=tag_to_ix["<PAD>"])
    return sentences_padded, tags_padded

# # 创建 DataLoader 实例时使用 collate_fn
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
ix_to_tag = {index: tag for tag, index in tag_to_ix.items()}
# 接下来就可以使用 train_loader 和 test_loader 在模型中进行训练和测试了




