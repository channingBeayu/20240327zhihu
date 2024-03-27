import numpy as np
import torch
from torch import nn, optim
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from torch.utils.data import TensorDataset, DataLoader
from utils.conn import Conn

y_data = []
topic_data = []
hc_data = []
doc_ids = []
with open('../data/sentis.txt', 'r') as file:
    for line in file:
        datas = line.split()
        y_data.append([float(datas[0])])
        topic_data.append([float(xi) for xi in datas[1:]])
with open('../data/hcs.txt', 'r') as file:
    for line in file:
        datas = line.split()
        doc_ids.append(int(datas[0]))
        hc_data.append([float(xi) for xi in datas[1:]])

y_data = torch.Tensor(y_data)
x_data = torch.Tensor(topic_data)
hc_data = torch.Tensor(hc_data)
doc_ids = torch.Tensor(doc_ids)

dataset = TensorDataset(x_data, y_data, hc_data, doc_ids)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [600, 245])
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

def pca(dataMat, topNfeat=4096):
    # 求矩阵每一列的均值
    meanVals = np.mean(np.asarray(dataMat), axis=0)
    # 数据矩阵每一列特征减去该列特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵，处以n-1是为了得到协方差的无偏估计
    # cov(x, 0) = cov(x)除数是n-1(n为样本个数)
    # cov(x, 1)除数是n
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值及对应的特征向量
    # 均保存在相应的矩阵中
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # sort():对特征值矩阵排序(由小到大)
    # argsort():对特征矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = np.argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[: -(topNfeat +1): -1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:, eigValInd]
    # 将去除均值后的矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    # 此处用转置和逆的结果一样redEigVects.I
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 返回压缩后的数据矩阵及该矩阵反构出原始数据矩阵
    # return lowDDataMat, reconMat
    return np.asarray(lowDDataMat)


class TextEmbedding(nn.Module):
    def __init__(self):
        super(TextEmbedding, self).__init__()
        # 初始化嵌入层，vocab_size 是词汇表大小，embed_dim 是嵌入向量的大小
        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        self.embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")

    def forward(self, text_input):
        embeddings = self.embedding_model.embed_documents(document=text_input)
        return embeddings


# 定义多层感知器层
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # 初始化MLP层
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        # x 应该是一个形状为 (batch_size, input_dim) 的 FloatTensor
        return torch.sigmoid(self.fc(x))

# 定义整个模型
class ThreeLayerModel(nn.Module):
    def __init__(self):
        super(ThreeLayerModel, self).__init__()
        self.embedding = TextEmbedding()
        # self.mlp = MLP(input_dim=25, hidden_dim=8, output_dim=1)
        self.mlp = MLP(input_dim=16, hidden_dim=8, output_dim=1)
        # self.mlp = MLP(input_dim=8, hidden_dim=8, output_dim=1)
        self.conn = Conn()
    def forward(self, topic_sentis, hc, doc_ids):
        # 文本混淆
        text_input = self.conn.get_docs(doc_ids)
        embed_output = self.embedding(text_input)
        embed_output = pca(embed_output, 8)
        # # 用户混淆
        # hc_output = pca(hc, 1)

        # 拼接后：k(8) + 16 + 1 = 25
        # combined_output = np.hstack((topic_sentis, embed_output, hc_output))
        combined_output = np.hstack((topic_sentis, embed_output))
        combined_output = torch.Tensor(combined_output)

        # combined_output = topic_sentis
        out_put = self.mlp(combined_output)
        return out_put


model = ThreeLayerModel()
for p in model.parameters():  # 初始化模型参数
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    for inputs, targets, hcs, doc_ids in train_loader:
        outputs = model(topic_sentis=inputs, hc=hcs, doc_ids=doc_ids)

        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
# 模型训练完成

# 测试模型
model.eval()
res = []
with torch.no_grad():
    for inputs, targets, hcs, doc_ids in test_loader:
        outputs = model(topic_sentis=inputs, hc=hcs, doc_ids=doc_ids)
        y_pred_cls = torch.round(torch.sigmoid(outputs))
        correct = (y_pred_cls == targets).sum().item()
        accuracy = correct / targets.size(0)
        print(f'Accuracy: {accuracy * 100} %')
        res.append(accuracy)
res = np.asarray(res).mean()
print(f'Mean Accuracy: {res * 100} %')




