import mysql.connector


class Conn:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root',
                                            db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()

    def get_docs(self, doc_ids):
        docs = []
        for doc_id in doc_ids:
            sql = "select content from 01docs_info where doc_id = %s"
            self.cursor.execute(sql, (int(doc_id),))
            docs.append(self.cursor.fetchone()[0])
        return docs

    def close(self):
        self.cursor.close()
        self.conn.close()