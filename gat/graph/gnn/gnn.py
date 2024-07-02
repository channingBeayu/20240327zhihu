import os.path as osp  # 导入os.path模块，并重命名为osp，用于处理文件路径

import torch  # 导入PyTorch库
from sklearn.metrics import roc_auc_score  # 从scikit-learn库导入roc_auc_score函数，用于评估模型

import torch_geometric.transforms as T  # 导入PyTorch Geometric的transforms模块，用于图数据的变换
from torch_geometric.datasets import Planetoid  # 导入PyTorch Geometric的Planetoid数据集
from torch_geometric.nn import GCNConv  # 导入PyTorch Geometric的GCNConv图卷积层
from torch_geometric.utils import negative_sampling  # 导入负采样工具函数

# 检测可用的设备，优先使用CUDA，其次是MPS（苹果的Metal Performance Shaders），最后是CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# 定义数据变换，包括特征归一化、设备转移和随机链接分割
transform = T.Compose([
    T.NormalizeFeatures(),  # 特征归一化
    T.ToDevice(device),  # 转移到指定设备
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),  # 随机链接分割
])
# 设置数据集路径并加载Planetoid数据集，这里以Cora数据集为例
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora', transform=transform)
# 数据集被分割为训练集、验证集和测试集
train_data, val_data, test_data = dataset[0]

# 定义图卷积网络模型
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)  # 第一层图卷积
        self.conv2 = GCNConv(hidden_channels, out_channels)  # 第二层图卷积

    def encode(self, x, edge_index):  # 编码函数，用于节点特征的转换
        x = self.conv1(x, edge_index).relu()  # 第一层卷积后使用ReLU激活函数
        return self.conv2(x, edge_index)  # 第二层卷积

    def decode(self, z, edge_label_index):  # 解码函数，用于预测边是否存在
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):  # 解码所有节点对的函数
        prob_adj = z @ z.t()  # 计算节点特征的内积作为边的预测概率
        return (prob_adj > 0).nonzero(as_tuple=False).t()  # 返回概率大于0的边

# 初始化模型、优化器和损失函数
model = Net(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# 定义训练函数
def train():
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空梯度
    z = model.encode(train_data.x, train_data.edge_index)  # 对训练数据进行编码

    # 对每个训练周期进行一次新的负采样
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    # 合并正样本和负样本的边索引
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    # 创建边的标签，正样本为1，负样本为0
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)  # 解码边的存在概率
    loss = criterion(out, edge_label)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
    return loss

# 定义测试函数，用于评估模型在验证集和测试集上的性能
@torch.no_grad()  # 不计算梯度，以节省计算资源
def teest(data):
    model.eval()  # 设置模型为评估模式
    z = model.encode(data.x, data.edge_index)  # 对数据进行编码
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()  # 解码并应用Sigmoid函数得到边存在的概率
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())  # 计算并返回ROC AUC分数

# 训练和评估模型
best_val_auc = final_test_auc = 0
for epoch in range(1, 101):  # 进行100个训练周期
    loss = train()  # 训练模型
    val_auc = teest(val_data)  # 在验证集上评估模型
    test_auc = teest(test_data)  # 在测试集上评估模型
    if val_auc > best_val_auc:  # 更新最佳验证集AUC和对应的测试集AUC
        best_val_auc = val_auc
        final_test_auc = test_auc
    # 打印训练信息
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

# 打印最终测试集上的AUC分数
print(f'Final Test: {final_test_auc:.4f}')

# 对测试集数据进行编码并解码所有节点对，用于预测图中所有可能的边
z = model.encode(test_data.x, test_data.edge_index)
final_edge_index = model.decode_all(z)
print()