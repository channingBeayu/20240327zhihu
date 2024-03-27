import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# 创建一个简单的逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=1):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# 准备数据集
# 这里我们创建一个简单的线性关系：y = 3 * x + 4 + 噪声
# 噪声是均值为0，标准差为0.5的正态分布
# x_data = torch.Tensor([[i] for i in range(100)])  # 生成一个线性递增的的特征数据
# y_data = torch.Tensor([3 * i + 4 + torch.randn(1) * 0.5 for i in range(100)])  # 生成目标数据

x_data = []
y_data = []
with open('../data/sentis.txt', 'r') as file:
    for line in file:
        datas = line.split()
        y_data.append([float(datas[0])])
        x_data.append([float(xi) for xi in datas[1:]])

x_data = torch.Tensor(x_data)
y_data = torch.Tensor(y_data)

# 转换为 PyTorch 的 DataLoader 格式

# 创建一个TensorDataset来存储数据
dataset = TensorDataset(x_data, y_data)
# 将数据分为训练集和测试集
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [600, 245])
# 使用DataLoader来简化数据加载过程
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

# 构建模型
input_size = 8
model = LogisticRegressionModel(input_size)

# 初始化模型参数
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# 定义损失函数
loss_function = nn.BCELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    for inputs, targets in train_loader:
        # 前向传播
        outputs = model(inputs)

        # 这里确保 targets 的尺寸是正确的
        # 如果输出层只有一个神经元，则不需要 unsqueeze
        # if outputs.size(1) > 1:
        #     targets = targets.unsqueeze(1)

        loss = loss_function(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

# 模型训练完成

# 测试模型
model.eval()  # 将模型设置为评估模式
res = []
with torch.no_grad():  # 在这个with下，所有计算得出的tensor都不会计算梯度，也就是不会进行反向传播
    for inputs, targets in test_loader:
        outputs = model(inputs)
        y_pred_cls = torch.round(torch.sigmoid(outputs))  # 将概率转换为类别标签
        correct = (y_pred_cls == targets).sum().item()
        accuracy = correct / targets.size(0)
        print(f'Accuracy: {accuracy * 100} %')
        res.append(accuracy)
res = np.asarray(res).mean()
print(f'Mean Accuracy: {res * 100} %')




