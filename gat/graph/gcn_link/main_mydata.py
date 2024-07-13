import os.path as osp

import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend

from model import Net

data = Data()

path = '../dataset/mydata/'
dataset = 'zh'
idx_features_labels = pd.read_csv("{}{}.content".format(path, dataset), sep='\t', header=None)
embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
data.x = torch.tensor(embedding_model.embed_documents(document=idx_features_labels[1]))
# data.num_nodes = data.x.shape[0]

edges_data = pd.read_csv("{}{}.cites".format(path, dataset), sep='\t', header=None).iloc[:, :2]
idx = idx_features_labels.iloc[:,0].values
idx_map = {j: i for i, j in enumerate(idx)}
edges = edges_data.iloc[:,0:2].applymap(lambda x:idx_map.get(x))
edges = edges.iloc[:,0:2].values.flatten('F').reshape(2, -1)  # # 将edges转为[2, edge]的形式
data.edge_index = torch.tensor(np.array(edges))


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
    z = model.encode(data.x, data.train_pos_edge_index)  # 输入节点和正样本边，输出(N*64) 应该是更新了节点特征
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # 输入节点、正样本边、负样本边，输出每条边存在的概率
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)  # 真实标签（111，000）
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    threshold = 0.5
    predicted_labels = torch.Tensor([1 if prob >= threshold else 0 for prob in link_logits])
    accuracy = sum(p_label == t_label for p_label, t_label in zip(predicted_labels, link_labels)) / len(link_labels)

    hamming_loss = sum(link_labels != predicted_labels) / len(link_labels)

    return loss, accuracy, hamming_loss

@torch.no_grad()
def teest(data, model):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

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

    # dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    # dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    # data = dataset[0]
    # ground_truth_edge_index = data.edge_index.to(device)
    # data.train_mask = data.val_mask = data.test_mask = data.y = None
    # data = train_test_split_edges(data)
    # data = data.to(device)

    # model = Net(dataset.num_features, 64).to(device)
    model = Net(data.x.shape[1], 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_auc = test_auc = 0
    for epoch in range(1, 101):
        loss, accuracy, hamming_loss = train(data, model, optimizer)
        val_auc, tmp_test_auc = teest(data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, acc: {accuracy:.4f}, hamming_loss: {hamming_loss:.4f},'
              f' Val: {val_auc:.4f}, Test: {test_auc:.4f}')

    z = model.encode(data.x, data.train_pos_edge_index)
    final_edge_index = model.decode_all(z)
    print()


if __name__ == "__main__":
    main()

