# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from torch_geometric.data import Data

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models import GAT_LP
from util import train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False, disjoint_train_ratio=0),
])

dataset = Planetoid(root_path + '/data', name='Cora', transform=transform)
train_data, val_data, test_data = dataset[0]

print(train_data)
print(val_data)
print(test_data)


data = Data()

path = 'F:/Code/240313/gat/graph/dataset/mydata/'
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


def main():
    # model = GAT_LP(dataset.num_features, 64, 128)
    # test_auc, test_ap = train(model, train_data, val_data, test_data, save_model_path=root_path + '/models/gat.pkl')
    model = GAT_LP(data.x.shape[1], 64, 128).to(device)
    test_auc, test_ap = train(model, data, data, data, save_model_path=root_path + '/models/gat.pkl')
    print('final best auc:', test_auc)
    print('final best ap:', test_ap)


if __name__ == '__main__':
    main()
