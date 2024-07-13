import os.path as osp
import re

import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from gat.graph.gat_node_effect.model import GAT as GAT_node

from gat.graph.gat_link.utils import load_data
from gat.graph.utils.get_weight import get_link_weight
from model import Net


data = Data()

path = '../dataset/mydata/'
dataset = 'zh'
idx_features_labels = pd.read_csv("{}{}.content".format(path, dataset), sep='\t', header=None)
# embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
# data.x = torch.tensor(embedding_model.embed_documents(document=idx_features_labels[1]))

###
import pickle
with open('../saved_model/node_features.pkl', 'rb') as f:
    a = pickle.loads(f.read())
data.x = a
###

# data.num_nodes = data.x.shape[0]

edges_data = pd.read_csv("{}{}.cites".format('../dataset/mydata/', 'zh'), sep='\t', header=None)
idx = idx_features_labels.iloc[:, 0].values
idx_map = {j: i for i, j in enumerate(idx)}
# 形成一个(N*N)的矩阵，[i,j]表示这条边的重要程度,N=features.shape[0]
link_weight = torch.ones(data.x.shape[0], data.x.shape[0])
for i in range(edges_data.shape[0]):
    row = edges_data.iloc[i]
    try:
        link_weight[idx_map[row[0]]][idx_map[row[1]]] = get_link_weight(row[2])
    except:
        print()
edges_data = edges_data.iloc[:, :2]
edges = edges_data.iloc[:,0:2].applymap(lambda x:idx_map.get(x))
edges = edges.iloc[:,0:2].values.flatten('F').reshape(2, -1)  # # 将edges转为[2, edge]的形式
data.edge_index = torch.tensor(np.array(edges))

# adj
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# print(data.edge_index.shape)
# torch.Size([2, 10556])

data = train_test_split_edges(data)



def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train(data, model, optimizer):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,  # 正样本
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))  # 负样本和正样本的数量一样多

    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index, adj, link_weight)  # 输入节点和正样本边，输出(N*64) 应该是更新了节点特征
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # 输入节点、正样本边、负样本边，输出每条边存在的概率
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)  # 真实标签（111，000）
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def teest(data, model):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index, adj, link_weight)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(data.x.shape[1], 64).to(device)

    ###
    seed = 2022
    dropout = 0.6
    alpha = 0.2
    nheads = 8
    hidden = 8
    lr = 0.005
    weight_decay = 5e-4
    model2 = GAT_node(nfeat=features.size()[1], nhid=hidden,
            nclass=labels.max().item() + 1, dropout=dropout,
            alpha=alpha, nheads=nheads)
    model2.load_state_dict(torch.load('../saved_model/model.pth'))
    model2.eval()
    nheads = 8
    for i in range(nheads):
        attention_module_name = f'attention_{i}'
        params_i = getattr(model2, attention_module_name).state_dict()
        getattr(model, attention_module_name).load_state_dict(params_i)
    ##

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_auc = test_auc = 0
    for epoch in range(1, 101):
        loss = train(data, model, optimizer)
        val_auc, tmp_test_auc = teest(data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    z = model.encode(data.x, data.train_pos_edge_index, adj, link_weight)
    final_edge_index = model.decode_all(z)
    print()

    # 保存模型
    # model_save_path = './model.pth'
    # torch.save(model.state_dict(), model_save_path)
    # print()


if __name__ == "__main__":
    main()

