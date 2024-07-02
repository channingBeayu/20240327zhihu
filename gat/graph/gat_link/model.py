import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class GAT(nn.Module):

    #GAT实现了多头注意力，这里需要指定头数nheads
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):

        super(GAT, self).__init__()
        self.name = 'gat'
        self.dropout = dropout
        #注意力部分
        #这里用生成式来表示多头注意力   1433*8 1433维降8维
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        #模型输出部分
        # GAT的输出层  64*1 64维变为1维（8个头的输出结果拼接成64维）
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)


    #GAT的计算方式      
    def forward(self, x, adj, pos_edge_index, neg_edge_index, link_weight):
        x = F.dropout(x, self.dropout, training=self.training)

        #计算并拼接由多头注意力所产生的特征矩阵
        x = torch.cat([att(x, adj, link_weight) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        #特征矩阵经由输出层得到最终的模型输出
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (x[edge_index[0]] * x[edge_index[1]]).sum(dim=-1)