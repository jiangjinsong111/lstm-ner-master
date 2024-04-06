import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datautils import NERDataset
from models import LSTMForNER
from datautils import train_data,word_to_ix,test_data,tag_to_ix,collate_fn,ix_to_tag
from sklearn.metrics import precision_recall_fscore_support

# 创建 DataLoader 实例时使用 collate_fn
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMForNER(len(word_to_ix), len(tag_to_ix), embedding_dim=64, hidden_dim=128)
model.to(device)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.15)

for epoch in range(10):  # 训练轮数
    model.train()
    total_loss = 0
    for sentence, tags in train_loader:
        sentence, tags = sentence.to(device), tags.to(device)
        model.zero_grad()
        tag_scores = model(sentence)
        loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), tags.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}")



# 假设我们有一批测试数据和相应的标签
# test_loader 是使用 DataLoader 创建的，它的 batch_size 设置为了 1 以简化处理
model.eval()  # 设置模型为评估模式

true_entities = []
pred_entities = []

with torch.no_grad():  # 在评估时不计算梯度
    for sentences, tags in test_loader:
        sentences = sentences.to(device)
        tags = tags.to(device)
        output = model(sentences)
        predictions = torch.argmax(output, dim=2)  # 获取最可能的标签索引

        # 将预测和真实标签转换为实体列表
        for true_tag, pred_tag in zip(tags.view(-1), predictions.view(-1)):
            if true_tag != tag_to_ix["<PAD>"]:  # 忽略填充的标签
                true_entities.append(ix_to_tag[true_tag.item()])
                pred_entities.append(ix_to_tag[pred_tag.item()])

# 计算精确率、召回率和F1分数
precision, recall, f1, _ = precision_recall_fscore_support(true_entities, pred_entities, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")



