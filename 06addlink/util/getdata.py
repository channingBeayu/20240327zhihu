from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from gat.graph.gat_link.utils import load_data
import pandas as pd
import numpy as np
import torch
import re
import pickle


def get_link_weight(link_label):
    type = re.findall(r'\[(.*?)\]', link_label)[0]
    if type == 'but':
        return 0.3
    elif type == 'causal':
        return 2.0
    elif type == 'condition':
        return 1.5
    elif type == 'more':
        return 1.0
    elif type == 'seq':
        return 2.0



def save_data():
    data = Data()

    path = 'F:/Code/240313/gat/graph/dataset/mydata/'
    dataset = 'zh'
    idx_features_labels = pd.read_csv("{}{}.content".format(path, dataset), sep='\t', header=None)
    embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
    data.x = torch.tensor(embedding_model.embed_documents(document=idx_features_labels[1]))

    ###
    # import pickle
    # with open('../saved_model/node_features.pkl', 'rb') as f:
    #     a = pickle.loads(f.read())
    # data.x = a
    edges_data = pd.read_csv("{}{}.cites".format(path, dataset), sep='\t', header=None)
    idx = idx_features_labels.iloc[:, 0].values
    idx_map = {j: i for i, j in enumerate(idx)}
    link_weight = torch.ones(data.x.shape[0], data.x.shape[0])
    for i in range(edges_data.shape[0]):
        row = edges_data.iloc[i]
        try:
            link_weight[idx_map[row[0]]][idx_map[row[1]]] = get_link_weight(row[2])
        except:
            print()
    edges_data = edges_data.iloc[:, :2]
    edges = edges_data.iloc[:,0:2].applymap(lambda x:idx_map.get(x))
    edges = edges.iloc[:,0:2].values.flatten('F').reshape(2, -1)
    data.edge_index = torch.tensor(np.array(edges))

    # adj
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path, dataset)
    data = train_test_split_edges(data)

    with open('../data/data.pkl', "wb") as f:
        pickle.dump(data, f)
    with open('../data/adj.pkl', "wb") as f:
        pickle.dump(adj, f)
    with open('../data/link_weight.pkl', "wb") as f:
        pickle.dump(link_weight, f)

    idx_map = {i: j for i, j in enumerate(idx)}
    with open('../data/idx_map.pkl', "wb") as f:
        pickle.dump(idx_map, f)

    print()

save_data()