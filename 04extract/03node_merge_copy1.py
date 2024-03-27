import mysql.connector
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

class NodeMerge:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root', db='paper', charset='utf8')
        self.cursor = self.conn.cursor()
        self.embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")

    def get_topic_nums(self):
        sql = "SELECT max(topic) FROM events"
        self.cursor.execute(sql)
        return int(self.cursor.fetchone()[0])

    # 1、取所有的part，构建节点id
    def get_data(self):
        sql = "select pre_part, post_part, pre_topic, post_topic, type, tag from events_copy1"
        self.cursor.execute(sql)
        items = self.cursor.fetchall()

        topics = []
        events = []
        labels = []
        for item in items:
            events.append(item[0])
            events.append(item[1])
            topics.append(int(item[2]))
            topics.append(int(item[3]))
            labels.append('[{}]{}'.format(item[4], item[5]))

        event_embeddings = self.embedding_model.embed_documents(document=events)
        return topics, events, event_embeddings, labels

    # 1、提取出同一聚类下的所有 事件节点 及其嵌入
    def get_topic_k(self, k):
        sql = "select event_id, pre_part, post_part, type, tag, pre_topic, post_topic from events_copy1"
        self.cursor.execute(sql, (k, ))
        items = self.cursor.fetchall()

        # 35个k归纳为8种类型
        k = get_new_k(k)


        events = []
        event_infos = []
        # pre_part用1表示，post_part用2表示
        for item in items:
            events.append(item[1])
            event_infos.append({'event_id': item[0],
                                'label': '[{}]{}'.format(item[3], item[4]),
                                'k': item[5]})

            events.append(item[2])
            event_infos.append({'event_id': item[0],
                                'label': '[{}]{}'.format(item[3], item[4]),
                                'k': item[6]})

        event_embeddings = self.embedding_model.embed_documents(document=events)
        return event_embeddings, event_infos, events

    # 2、计算相似度，并合并节点
    def event_merge(self, event_embeddings, events, topics):

        for k in range(topic_nums):
            events = np.asarray(events)
            topics = np.asarray(topics)

            # 聚类k的事件和向量
            events_k = events[topics == k]
            event_embeddings_k = event_embeddings[topics == k]

            print('****' * 6)
            print('{}聚类下，两两一对看相似度'.format(k))
            sim = cosine_similarity(event_embeddings[topics == k])
            print_paired_events(sim, events)  # 可以打印一下，
            for i in range(len(events_k) - 2):
                if i % 2 == 0:
                    idx = np.argmax(sim[i][i + 2:]) + i + 2
                else:
                    idx = np.argmax(sim[i][i + 1:]) + i + 1
                if sim[i][idx] > 0.5:
                    events[idx] = events[i]  # TODO: 想一下，是直接覆盖后面的句子 还是怎么着
            events[topics == k] = events_k


        return events

    def to_graph(self, events, topics, labels):
        nodes = []
        edges = []

        i = 0
        while i < len(events):
            nodes.append(events[i])
            nodes.append(events[i+1])
            edges.append({'from': events[i], 'to': events[i+1],
                          'label': labels[int(i/2)]})
            i = i+2

        # node_dict = {node: {'index': index, 'k': event_infos} for index, node in enumerate(nodes)}
        node_dict = {}
        for index, node in enumerate(nodes):
            node_dict[node] = {'index': index, 'k': get_new_k(topics[index])}


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

        f = open('travel_event_graph.html', 'w+', encoding='utf-8')
        base = '''
            <html>
            <head>
              <script type="text/javascript" src="VIS/dist/vis.js"></script>
              <link href="VIS/dist/vis.css" rel="stylesheet" type="text/css">
              <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
            </head>
            <body>

            <div id="VIS_draw"></div>
            <script type="text/javascript">
              var nodes = data_nodes;
              var edges = data_edges;

              var container = document.getElementById("VIS_draw");

              var data = {
                nodes: nodes,
                edges: edges
              };

              var options = {
                  nodes: {
                      shape: 'dot',
                      size: 25,
                      font: {
                          size: 14
                      }
                  },
                  edges: {
                      font: {
                          size: 14,
                          align: 'middle'
                      },
                      color: 'gray',
                      arrows: {
                          to: {enabled: true, scaleFactor: 0.5}
                      },
                      smooth: {enabled: false}
                  },
                  physics: {
                      enabled: true
                  }
              };

              var network = new vis.Network(container, data, options);

            </script>
            </body>
            </html>
            '''
        html = base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges))
        f.write(html)
        f.close()


merge = NodeMerge()
topic_nums = merge.get_topic_nums()
eventss = []
event_infoss = []

topics, events, event_embeddings, labels = merge.get_data()
events = merge.event_merge(event_embeddings, events, topics)
merge.to_graph(events, topics, labels)


# for k in range(topic_nums):  # range(topic_nums)
#     event_embeddings, event_infos, events = merge.get_topic_k(k)
#     event_infoss.extend(event_infos)
#     events = merge.node_merge(event_embeddings, events)
#     eventss.extend(events)
# merge.to_graph(eventss, event_infoss)




# 前置定语和主语(ATT)一致





