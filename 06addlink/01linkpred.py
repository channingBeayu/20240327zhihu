import os.path as osp
import pickle
import torch

from gat.graph.gcn_gat_link_weight_node_b.model import Net

# 1、加载模型输入数据
with open('data/data.pkl', "rb+") as f:
    data = pickle.load(f)
with open('data/adj.pkl', "rb+") as f:
    adj = pickle.load(f)
with open('data/link_weight.pkl', "rb+") as f:
    link_weight = pickle.load(f)


# 2、模型预测
model = Net(data.x.shape[1], 64)
model.load_state_dict(torch.load('data/model.pth'))
model.eval()

z = model.encode(data.x, data.train_pos_edge_index, adj, link_weight)  # (1176, 64)
final_edge_index = model.decode_all(z)  # 249

# 3、id转换，并加入边集合
with open(r'F:\Code\240313\gat\graph\dataset\mydata\data_edges.pkl', "rb+") as f:
    data_edges = pickle.load(f)
with open(r'F:\Code\240313\gat\graph\dataset\mydata\data_nodes.pkl', "rb+") as f:
    data_nodes = pickle.load(f)
# data_edges的from和to是节点真实id（>2000）
# 所以要把预测的索引id转为真实节点id，然后加到data_edges中
with open('data/idx_map.pkl', "rb+") as f:
    idx_map = pickle.load(f)

for i in range(final_edge_index.shape[1]):
    edge = final_edge_index[:, i]
    data_edges.append({'from': idx_map[int(edge[0])],
                       'to': idx_map[int(edge[1])],
                       'label': 'pred'})

# 4、保存边集合
with open('data/data_edges.pkl', "wb") as f:
    pickle.dump(data_edges, f)

print()


