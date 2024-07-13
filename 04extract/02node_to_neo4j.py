import pickle
import re

import mysql.connector
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from py2neo import Graph, Subgraph, NodeMatcher
from py2neo import Node, Relationship, Path

def print_paired_events(sim, events):
    pairs = []
    for i in range(len(sim) - 1):
        for j in range(i + 1, len(sim)):
            pairs.append({'index': [i, j], 'score': sim[i][j]})
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    for pair in pairs:
        i, j = pair['index']
        print("{:<30} \t\t {:<30} \t\t Score: {:.4f}".format(events[i], events[j], pair['score']))


def count_events(events, event_infos):
    count_events = {}
    for index, string in enumerate(events):
        if string not in count_events:
            count_events[string] = {'count': 1, 'first_index': index}
        else:
            count_events[string]['count'] += 1

    data = []
    # 组合一下，形成元组（event_id, pre/post, event, count）  PS:统计个数好像目前没啥用
    for key, value in count_events.items():
        data.append((event_infos[value['first_index']][0], event_infos[value['first_index']][1],
                     key, value['count']))
        print(f"键：{key}，值：{value}")

    return data


def get_effect(k):
    effect = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, ]
    return effect[k]


def get_relation_weight(relation):
    dict = {'causal': 1, 'condition': 1,
            'more': 0.5, 'seq': 0.5,
            'but': -0.5}
    if relation in dict:
        return dict[relation]
    else:
        return dict[re.findall(r'\[(.*?)\]', relation)[0]]


class NodeMerge:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root', db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()
        self.embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
        self.graph = Graph('bolt://localhost:7687', auth=('neo4j', '123456'))

        with open('utils/topic_model.pkl', 'rb') as f:
            self.topic_model = pickle.loads(f.read())
        # BerTopic_model = GBERTopic.load("my_topics_model")


    # 1、提取出同一聚类下的所有 事件节点 及其嵌入
    def get_topic_k(self, k):
        sql = "select event_id, pre_part, post_part, type, tag from events where doc_id = %s"
        self.cursor.execute(sql, (k, ))
        items = self.cursor.fetchall()

        events = []
        event_infos = []
        # pre_part用1表示，post_part用2表示
        for item in items:
            events.append(item[1])
            event_infos.append({'event_id': item[0], 'pre_or_post': 1,
                                'label': item[3],
                                'k': k})

            events.append(item[2])
            event_infos.append({'event_id': item[0], 'pre_or_post': 2,
                                'label': '[{}]{}'.format(item[3], item[4]),
                                'k': k})

        event_embeddings = self.embedding_model.embed_documents(document=events)
        return event_embeddings, event_infos, events


    # 1、提取出同一聚类下的所有 事件节点 及其嵌入
    def get_topic(self,):
        sql = "select doc_id, pre_part, post_part, type, tag from events"
        self.cursor.execute(sql)
        items = self.cursor.fetchall()

        events = []
        event_infos = []
        # pre_part用1表示，post_part用2表示
        for item in items:
            events.append(item[1])
            event_infos.append({'k': item[0], 'pre_or_post': 1,
                                'label': item[3],})

            events.append(item[2])
            event_infos.append({'k': item[0], 'pre_or_post': 2,
                                'label': '[{}]{}'.format(item[3], item[4]),})

        event_embeddings = self.embedding_model.embed_documents(document=events)
        return event_embeddings, event_infos, events


    # 2、计算相似度，并合并节点
    def node_merge(self, event_embeddings, events):
        sim = cosine_similarity(event_embeddings)
        print('****' * 6)
        # print('{}聚类下，两两一对看相似度'.format(k))
        # print_paired_events(sim, events)  # 可以打印一下，但是有点慢
        for i in range(len(events) - 2):
            if i % 2 == 0:
                idx = np.argmax(sim[i][i + 2:]) + i + 2
            else:
                idx = np.argmax(sim[i][i + 1:]) + i + 1
            if sim[i][idx] > 0.5:
                events[idx] = events[i]    # TODO: 想一下，是直接覆盖后面的句子 还是怎么着
        return events

    def to_graph(self, events, event_infos):
        nodes = []
        edges = []

        i = 0
        while i < len(events):
            nodes.append(events[i])
            nodes.append(events[i+1])
            edges.append({'from': events[i], 'to': events[i+1],
                          'from_k': event_infos[i]['k'], 'to_k': event_infos[i+1]['k'],
                          'label': event_infos[i]['label']})
            i = i+2

        node_dict = {}
        for index, node in enumerate(nodes):
            if node not in node_dict:
                # node_dict[node] = {'index': index, 'k': event_infos[index]['k']}
                node_dict[node] = {'k': event_infos[index]['k']}

        for node, info in node_dict.items():
            neo_node = Node(str(info['k']), name=node)
            self.graph.create(neo_node)

        node_matcher = NodeMatcher(self.graph)  # 节点匹配器
        for index, edge in enumerate(edges):
            # 1. 组装边的权重
            # 计算效应值，即from节点的类别效应
            effect = get_effect(edge['from_k'])
            # 获取句间关系的权重
            relation_weight = get_relation_weight(event_infos[index]['label'])
            # 边的值： 0.6 * effect + 0.4 * relation
            weight = 0.6 * effect + 0.4 * relation_weight

            # 2. 查找图中的节点 并建边
            from_node = node_matcher.match(str(edge['from_k']), name=edge['from']).first()
            to_node = node_matcher.match(str(edge['to_k']), name=edge['to']).first()
            if from_node and to_node:
                relation = Relationship(from_node, str(weight), to_node, weight=-weight)
                self.graph.create(relation)

merge = NodeMerge()
topic_nums = 2  # 8
event_embeddings, event_infos, events = merge.get_topic()
eventss = merge.node_merge(event_embeddings, events)
merge.to_graph(eventss, event_infos)



