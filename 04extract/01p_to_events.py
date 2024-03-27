import mysql.connector

from utils import GetEvents


class Doc2Events:
    def __init__(self):
        self.extractor = GetEvents()
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root',
                                            db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()

    # 1、从数据库获取所有docs
    def get_ps(self):
        sql = "select p, topic from 01docs_p " # limit 1000
        self.cursor.execute(sql)
        self.ps_info = self.cursor.fetchall()

    # 2、对文章提取事件，并按字段顺序组成为元组的形式
    def extract_events(self, p):
        events = self.extractor.get_events(p)
        res = []
        for event in events:
            res.append([event['sent'], event['type'],event['tag'],
                        event['tuples']['pre_part'], event['tuples']['post_part']])
        return res

    # 3、存储事件
    def save_events(self, events):
        sql = "insert into events(doc_id, sent, type, tag, pre_part, post_part)" \
              "values (%s, %s, %s, %s, %s, %s)"
        self.cursor.executemany(sql, events)
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def extract_main(self):
        self.get_ps()
        events = []
        for p_info in self.ps_info:
            eventt = self.extract_events(p_info[0])
            for event in eventt:
                events.append((p_info[1], *event))
        # 10个文章就421个事件对了，而且分句质量有好有坏
        self.save_events(events)


Doc2Events().extract_main()


