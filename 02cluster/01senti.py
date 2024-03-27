import re

import mysql.connector
from snownlp import SnowNLP
from snownlp import sentiment

class Senti:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root',
                                            db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()

    def main(self):
        infos = self.get_contents()
        res = self.senti(infos)
        self.save('01docs_info', 'doc_id', res)

        infos = self.get_ps()
        res = self.senti(infos)
        self.save('01docs_p', 'p_id', res)

        self.close()

    def senti(self, infos):
        res = []
        for info in infos:
            doc = info[1]
            doc = re.sub(r'[\s\r\n]|<n/>', "", doc)
            if doc:  #
                senti = SnowNLP(doc).sentiments
            else:
                senti = 0
            print(f'{doc}的情感得分为：{senti}')
            res.append((senti, info[0]))
        return res

    def get_contents(self):
        sql = "select doc_id, content from 01docs_info"
        self.cursor.execute(sql)
        doc_infos = self.cursor.fetchall()
        return doc_infos

    def get_ps(self):
        sql = "select p_id, p from 01docs_p"
        self.cursor.execute(sql)
        p_infos = self.cursor.fetchall()
        return p_infos

    def save(self, table_name, id_name, data):
        update_query = f"UPDATE {table_name} SET sentiment = %s WHERE {id_name} = %s"
        self.cursor.executemany(update_query, data)
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

Senti().main()

