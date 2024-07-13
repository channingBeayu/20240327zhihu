import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from gat.graph.gcn_gat_link_weight_node_b.gat_layers import GraphAttentionLayer


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.dropout = 0.6

        #注意力部分 1433维降8维
        #这里用生成式来表示多头注意力   1433*8
        self.attentions = [GraphAttentionLayer(in_channels, 8, dropout=0.6, alpha=0.2, concat=True) for _ in range(8)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        self.conv1 = GCNConv(64, 8)
        self.conv2 = GCNConv(8, out_channels)

    def encode(self, x, edge_index, adj, link_weight):
        x = torch.cat([att(x, adj, link_weight) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # 2行，后面拼接负样本
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        # 括号里是(边的总条数，64维)
        # z[edge_index[0]] 是(边的总条数，64维)
        # 返回值是(边的总条数)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        # return (prob_adj > 0).nonzero(as_tuple=False).t()

        links = (prob_adj > 1.7).nonzero(as_tuple=False).t()
        # 删除同一个节点的环
        duplicate_columns = []
        for i in range(links.shape[1]):
            column = links[:, i]
            if column[0] == column[1]:
                duplicate_columns.append(i)

        links = links[:, [index for index in range(links.shape[1]) if index not in duplicate_columns]]
        return links

