import torch
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, 8)
        self.conv2 = GCNConv(8, out_channels)

    def encode(self, x, edge_index):
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
        return (prob_adj > 0).nonzero(as_tuple=False).t()

