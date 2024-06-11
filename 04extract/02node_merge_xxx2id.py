import mysql.connector
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

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

def get_relation2id():
    with open('data/relation2id.txt', 'w', encoding='utf-8') as file:
        file.write(str(5) + '\n')
        file.write('{}\t{}\n'.format('but', 0))
        file.write('{}\t{}\n'.format('causal', 1))
        file.write('{}\t{}\n'.format('condition', 2))
        file.write('{}\t{}\n'.format('more', 3))
        file.write('{}\t{}\n'.format('seq', 4))

def get_entity2id(data_nodes):
    with open('data/entity2id.txt', 'w', encoding='utf-8') as file:
        file.write(str(len(data_nodes)) + '\n')
        for entity in data_nodes:
            # file.write(entity['label'] + '\t' + entity['id'] + '\n')
            file.write('{}\t{}\n'.format(entity['label'], entity['id']))


def get_train2id(data_edges):
    with open('data/train2id.txt', 'w', encoding='utf-8') as file:
        file.write(str(len(data_edges)) + '\n')
        for relation in data_edges:
            # file.write(relation['from'] + '\t' + relation['to'] + '\t' + getRid(relation['label'])+ '\n')
            file.write('{}\t{}\t{}\n'.format(relation['from'], relation['to'], getRid(relation['label'])))


def getRid(label):
    type = re.findall(r'\[(.*?)\]', label)[0]
    if type == 'but':
        return "0"
    elif type == 'causal':
        return "1"
    elif type == 'condition':
        return "2"
    elif type == 'more':
        return "3"
    elif type == 'seq':
        return "4"

class NodeMerge:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root', db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()
        self.embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")


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
                                'label': '[{}]{}'.format(item[3], item[4]),
                                'k': k})

            events.append(item[2])
            event_infos.append({'event_id': item[0], 'pre_or_post': 2,
                                'label': '[{}]{}'.format(item[3], item[4]),
                                'k': k})

        event_embeddings = self.embedding_model.embed_documents(document=events)
        return event_embeddings, event_infos, events


    # 2、计算相似度，并合并节点
    def node_merge(self, event_embeddings, events):
        sim = cosine_similarity(event_embeddings)
        print('****' * 6)
        print('{}聚类下，两两一对看相似度'.format(k))
        print_paired_events(sim, events)  # 可以打印一下，
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
                          'label': event_infos[i]['label']})
            i = i+2

        # node_dict = {node: {'index': index, 'k': event_infos} for index, node in enumerate(nodes)}
        node_dict = {}
        index = 0
        for node in nodes:
            if node not in node_dict:
                node_dict[node] = {'index': index, 'k': event_infos[index]['k']}
                index += 1

        data_nodes = []
        data_edges = []
        for node, info in node_dict.items():
            data = {}
            data["group"] = info['k']
            data["id"] = info['index']
            data["label"] = node
            data_nodes.append(data)

        for index, edge in enumerate(edges):
            data = {}
            data['from'] = int(node_dict[edge['from']]['index'])
            data['label'] = edge['label']
            data['to'] = int(node_dict[edge['to']]['index'])
            data_edges.append(data)

        # f = open('travel_event_graph.html', 'w+', encoding='utf-8')
        # html = base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges))
        get_entity2id(data_nodes)
        get_train2id(data_edges)
        get_relation2id()

# 聚类内部合并的版本
merge = NodeMerge()
topic_nums = 1
eventss = []
event_infoss = []
for k in range(4):
    event_embeddings, event_infos, events = merge.get_topic_k(k)
    event_infoss.extend(event_infos)
    events = merge.node_merge(event_embeddings, events)
    eventss.extend(events)
merge.to_graph(eventss, event_infoss)

# 全部聚类一起计算相似度的版本
# eventss = []
# event_infoss = []
# event_embeddingss = []
# for k in range(7):
#     event_embeddings, event_infos, events = merge.get_topic_k(k)
#     event_embeddingss.extend(event_embeddings)
#     event_infoss.extend(event_infos)
#     eventss.extend(events)
# eventss = merge.node_merge(event_embeddingss, eventss)
# merge.to_graph(eventss, event_infoss)


