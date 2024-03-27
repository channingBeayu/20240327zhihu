import mysql.connector
import numpy as np

class GetData:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root',
                                            db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()

    def main(self):
        ys = self.get_doc_sentis()

        data = []
        data_hc = []
        for doc_info in ys:
            sql = "select topic, sentiment from 01docs_p where doc_id = %s"
            self.cursor.execute(sql, (doc_info[0],))
            ps = self.cursor.fetchall()
            if not ps:
                continue

            sentis = np.asarray(0).repeat(k)
            count = np.asarray(0).repeat(k)
            for p_topic, p_sentiment in ps:
                sentis[p_topic] = p_sentiment
                count[p_topic] += 1
            row = [doc_info[1]]
            row.extend([round(sentis[idx] / count[idx], 2) if count[idx] > 0 else 0.5 for idx in range(k)])
            data.append(row)

            row_hc = [doc_info[0]]
            row_hc.extend(doc_info[2:])
            data_hc.append(row_hc)
        self.close()

        # with open('data/sentis.txt', 'w') as f:
        #     for row in data:
        #         f.write(' '.join([str(item) for item in row]) + '\n')

        with open('data/hcs.txt', 'w') as f:
            for row in data_hc:
                f.write(' '.join([str(item) if str(item) != '' else '0' for item in row]) + '\n')

    def get_doc_sentis(self):
        sql = "select doc_id, sentiment, fans_count, comment_count, voteup_count from 01docs_info"
        self.cursor.execute(sql)
        ys = self.cursor.fetchall()
        return ys

    def close(self):
        self.cursor.close()
        self.conn.close()

k = 8
GetData().main()
