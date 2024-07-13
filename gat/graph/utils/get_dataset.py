
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from gat.graph.gat_link.utils import load_data, load_data_cora
import torch

def get_data_zh():
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

    # adj
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # print(data.edge_index.shape)
    # torch.Size([2, 10556])

    data = train_test_split_edges(data)
    return data, adj


def get_data_cora():
    path = '../dataset/cora/'
    dataset = 'cora'
    data = Data()
    adj, features, labels, idx_train, idx_val, idx_test = load_data_cora()

    idx_features_labels = pd.read_csv("{}{}.content".format(path, dataset), sep='\t', header=None)
    embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
    data.x = features
    # data.num_nodes = data.x.shape[0]

    edges_data = pd.read_csv("{}{}.cites".format(path, dataset), sep='\t', header=None).iloc[:, :2]
    idx = idx_features_labels.iloc[:, 0].values
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = edges_data.iloc[:, 0:2].applymap(lambda x: idx_map.get(x))
    edges = edges.iloc[:, 0:2].values.flatten('F').reshape(2, -1)  # # 将edges转为[2, edge]的形式
    data.edge_index = torch.tensor(np.array(edges))

    data = train_test_split_edges(data)
    return data, adj