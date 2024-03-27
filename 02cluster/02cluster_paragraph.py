import pickle

import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import norm
from gbertopic.bertopic._bertopic_with_topicinfo import BERTopic as GBertopic
from gbertopic.bertopic.backend._sentencetransformers import SentenceTransformerBackend
from gbertopic.gsdmm.get_gsdmm import train_zh
from utils.ltp import LtpParser

def calculate_sse(data, labels):
    """
    计算SSE（Sum of Squared Errors）。
    :param data: numpy数组，形状为[n_samples, n_features]，表示数据集。
    :param labels: numpy数组，形状为[n_samples,]，表示每个样本的聚类标签。
    :param centers: numpy数组，形状为[n_clusters, n_features]，表示聚类中心。
    :return: SSE的值。 越小越好
    """
    labels = np.array(labels)
    centers = ''
    for i in range(len(np.unique(labels))-1):
        center = np.mean(data[labels == i], axis=0)
        if centers == '':
            centers = center
        else:
            centers = np.vstack((centers, center))

    sse = 0
    for i in range(len(centers)):
        # 获取属于当前聚类中心的样本
        cluster_data = data[labels == i]
        # 计算到聚类中心的距离平方和
        sse += np.sum(np.square(cluster_data - centers[i]))
    return sse


def silhouette_coefficient( x, label):
    # turn label into restrctly continuous (begin from 0)
    x = np.asarray(x)
    unique, label = np.unique(label, return_inverse=True)
    if unique.size == 1:
        raise ValueError('unique label should be > 2')
    assert x.shape[0] == label.size

    # distance between every two sample
    dist_matrix = norm(x[:, None, :] - x[None], axis=2)

    # distance between sample and cluster
    dist_matrix = np.asarray([np.ma.masked_equal(dist_matrix[:, label == i], 0).mean(1)
                              for i in range(len(unique))]).T

    # index of intra-distance
    idx = np.eye(len(unique), dtype=bool)[label]

    # result
    intra = dist_matrix[idx]
    inter = dist_matrix[~idx].reshape(label.size, -1)
    if inter.ndim > 1:
        inter = inter.min(1)
    sil = (inter - intra) / (np.maximum(intra, inter) + 1e-16)  # lest devided by 0
    sil[intra == 0] = 0  # when intra = 0, silhouette score should be 0

    # print(sil.mean())
    # return sil.mean(), sil
    return sil.mean()

def DaviesBouldin(X, labels, n_cluster):
    # 越低越好
    """
    计算DB指数
    :param X: 数据集
    :param labels: 聚类标签
    :return: 返回DBI
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    # n_cluster = len(np.bincount(labels))  # 簇的个数
    n_cluster = np.max(labels) + 1
    cluster_k = [X[labels == k] for k in range(n_cluster)]  # 对应簇的数据集
    centroids = [np.mean(k, axis=0) for k in cluster_k]  # 按照标签列，已经分好了集群，那么每个集群的数据的平均值就是质心所在的位置
    # 求S 簇类中数据到质心欧氏距离和的平均距离
    S = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    Ri = []
    for i in range(n_cluster):
        Rij = []
        # 计算Rij
        for j in range(n_cluster):
            if j != i:
                r = (S[i] + S[j]) / euclidean(centroids[i], centroids[j])
                Rij.append(r)
        # 求Ri
        Ri.append(max(Rij))
    # 求dbi
    dbi = np.mean(Ri)
    # print(dbi)
    return dbi

def euclidean(vec1, vec2):
        """
        闵可夫斯基距离
        :param vec1:样本 1
        :param vec2:样本 2
        :param p:p=1：曼哈顿距离；p=2: 欧氏距离
        :return:返回所求距离值
        """
        vec = []
        p = 2
        i = 0
        sum = 0
        while i < len(vec1):
            # vec[i] = abs(vec1[i] - vec2[i]) ** p
            sum += abs(vec1[i] - vec2[i]) ** p
            i += 1
        if p == 1:
            return sum
        elif p == 2:
            return sum ** 0.5


class ClusterNum:
    def __init__(self):
        self.embedding_model = SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root',
                                            db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()
        self.ltp = LtpParser()

    def get_ps(self):
        sql = "select p_id, p from 01docs_p"
        self.cursor.execute(sql)
        res = self.cursor.fetchall()
        self.p_id = []
        self.ps = []
        for item in res:
            self.p_id.append(item[0])
            self.ps.append(item[1])

    def get_curve(self):
        embeddings = self.embedding_model.embed_documents(document=self.ps)
        cut_ps = [self.ltp.doc_cut(p) for p in self.ps]

        x = range(3, 20)  # 100
        # y = [self.get_indicator_by_k(k, embeddings, cut_ps) for k in x]
        y = []
        for k in x:
            y.append(self.get_indicator_by_k(k, embeddings, cut_ps))
        plt.plot(x, y)
        plt.xlabel('主题数目')
        plt.ylabel('SSE')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.title('主题-SSE变化情况')
        plt.show()

    def get_indicator_by_k(self, k, embeddings, cut_ps):
        gsdmm_y, gsdmm_p = train_zh(cut_ps, k)
        topic_model = GBertopic(calculate_probabilities=True, nr_topics=k)
        predictions, probabilities = topic_model.fit_transform(self.ps, gsdmm_p=gsdmm_p, embeddings=embeddings)

        # indicator = self.silhouette_coefficient(embeddings, predictions)
        # print('轮廓系数：', indicator)

        # indicator = self.DaviesBouldin(embeddings, predictions, k-1)
        # print('DBi：', indicator)

        indicator = calculate_sse(embeddings, predictions)
        print(f'k: {k}, sse: {indicator}')
        print('---------------------------------')

        return indicator

    def main_cluster(self, k):
        embeddings = self.embedding_model.embed_documents(document=self.ps)

        cut_ps = [self.ltp.doc_cut(p) for p in self.ps]
        gsdmm_y, gsdmm_p = train_zh(cut_ps, k)
        topic_model = GBertopic(calculate_probabilities=True, nr_topics=k, language='chinese (simplified)')
        predictions, probabilities = topic_model.fit_transform(self.ps, gsdmm_p=gsdmm_p, embeddings=embeddings)

        print(topic_model.get_topics())
        # 保存模型和预测值
        topic_model.save("save/topic_gmodel")
        # with open('save/topic_model.pkl', 'wb') as f:
        #     pickle.dump(topic_model, f)
        # with open('save/predictions.pkl', 'wb') as f:
        #     pickle.dump(predictions, f)
        return predictions

    def set_topic(self, predictions):
        p_topic = []

        for id, topic in zip(self.p_id, predictions):
            p_topic.append((topic, id))

        table_name = '01docs_p'
        sql = f"UPDATE {table_name} SET topic = %s WHERE p_id = %s"
        self.cursor.executemany(sql, p_topic)
        self.conn.commit()

        self.cursor.close()
        self.conn.close()


cluster = ClusterNum()
cluster.get_ps()
cluster.get_curve()
# predictions = cluster.main_cluster(k=9)
# cluster.set_topic(predictions)


# 加载主题模型
# with open('save/topic_model.pickle', 'rb') as f:
#    topic_model = pickle.load(f)

