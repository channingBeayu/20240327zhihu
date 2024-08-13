import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import negative_sampling
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from gat.graph.utils.get_weight import get_link_weight
from model import GAT
from gat.graph.gat_node.model import GAT as GAT_node
from utils import *

import time



# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
num_epochs = 100  # 1000
seed = 2022
dropout = 0.6
alpha = 0.2
nheads = 8
hidden = 8
lr = 0.005
weight_decay = 5e-4
fix_seed(seed)

# data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
# adj, features, labels, idx_train, idx_val, idx_test = load_data_cora()

# link
path = '../dataset/mydata/'
dataset = 'zh'
# path = '../dataset/cora/'
# dataset = 'cora'
idx_features_labels = pd.read_csv("{}{}.content".format(path, dataset), sep='\t', header=None)
embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
edges_data = pd.read_csv("{}{}.cites".format(path, dataset), sep='\t', header=None)
idx = idx_features_labels.iloc[:, 0].values
idx_map = {j: i for i, j in enumerate(idx)}
# 形成一个(N*N)的矩阵，[i,j]表示这条边的重要程度,N=features.shape[0]
link_weight = torch.ones(features.shape[0], features.shape[0])
for i in range(edges_data.shape[0]):
    row = edges_data.iloc[i]
    try:
        link_weight[idx_map[row[0]]][idx_map[row[1]]] = get_link_weight(row[2])
    except:
        print()
edges_data = edges_data.iloc[:, :2]
edges = edges_data.iloc[:, 0:2].applymap(lambda x: idx_map.get(x))
edges = edges.iloc[:, 0:2].values.flatten('F').reshape(2, -1)  # # 将edges转为[2, edge]的形式
pos_edge_index = torch.tensor(edges)

model = GAT(nfeat=features.size()[1], nhid=hidden,
            nclass=2, dropout=dropout,
            alpha=alpha, nheads=nheads).to(device)
# # 将gat_node的前半部分的参数复制过来
# model2 = GAT_node(nfeat=features.size()[1], nhid=hidden,
#             nclass=labels.max().item() + 1, dropout=dropout,
#             alpha=alpha, nheads=nheads).to(device)
# model2.load_state_dict(torch.load('../saved_model/model.pth'))
# model2.eval()
# for i in range(nheads):
#     attention_module_name = f'attention_{i}'
#     params_i = getattr(model2, attention_module_name).state_dict()
#     getattr(model2, attention_module_name).load_state_dict(params_i)


# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(epoch):
    t = time.time()
    model.train()

    neg_edge_index = negative_sampling(  # 负样本
        edge_index=pos_edge_index,  # 正样本
        num_nodes=features.shape[0],
        num_neg_samples=pos_edge_index.shape[1])

    # Forward pass
    link_logits = model(features, adj, pos_edge_index, neg_edge_index, link_weight)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred = [1 if prob > 0.5 else 0 for prob in link_logits]
    print(
        'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.2f}, hamming_loss: {:.4f}, f1: {:.4f}, time: {:.4f}s'.
        format(
            epoch + 1, num_epochs, loss.item(),
            100 * accuracy_score(link_labels, y_pred),  # acc
            sum(link_labels != torch.Tensor(y_pred)) / len(link_labels),  # hamming_loss
            f1_score(link_labels, y_pred, average='weighted'),  # f1
            time.time() - t)
    )


def teest():
    with torch.no_grad():
        model.eval()

        outputs = model(features, adj)
        test_acc = accuracy(outputs[idx_test], labels[idx_test])
        print('test acc: {:.2f}'.format(test_acc * 100))
        # visualize(outputs, color=labels, model_name=model.name)
        print('tSNE image is generated')




def valid():
    with torch.no_grad():
        model.eval()
        outputs = model(features, adj)
        val_acc = accuracy(outputs[idx_val], labels[idx_val])
        print('val acc: {:.2f}'.format(val_acc * 100))
        # visualize(outputs, color=labels, model_name=model.name)
        print('tSNE image is generated')


# Train model 
t_total = time.time()
for epoch in range(num_epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
teest()
