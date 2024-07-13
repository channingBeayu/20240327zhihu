import pickle
import re

import mysql.connector
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from py2neo import Graph, Subgraph, NodeMatcher
from py2neo import Node, Relationship, Path

from py2neo import Graph, NodeMatcher
from py2neo import Node, Relationship



# 建立节点真实id和label之间的关系
def get_label_by_id(data_nodes):
    id_map = {}
    for node in data_nodes:
        id_map[node['id']] = node['label']
    return id_map



with open(r'F:\Code\240313\gat\graph\dataset\mydata\data_nodes.pkl', "rb+") as f:
    data_nodes = pickle.load(f)
id_map = get_label_by_id(data_nodes)
with open('data/data_edges.pkl', "rb+") as f:
    data_edges = pickle.load(f)

graph = Graph('bolt://localhost:7687', auth=('neo4j', '123456'))
graph.delete_all()

# 建节点
for node in data_nodes:
    neo_node = Node(str(node['group']), name=node['label'])  # 类别 name=事件
    graph.create(neo_node)

# 建边
node_matcher = NodeMatcher(graph)
for edge in data_edges:
    # 2. 查找图中的节点 并建边
    # from_node = node_matcher.match(str(edge['from_k']), name=edge['from']).first()
    from_node = node_matcher.match(name=id_map[edge['from']]).first()
    to_node = node_matcher.match(name=id_map[edge['to']]).first()
    if from_node and to_node:
        # relation = Relationship(from_node, str(weight), to_node, weight=-weight)
        relation = Relationship(from_node, edge['label'], to_node)
        graph.create(relation)




