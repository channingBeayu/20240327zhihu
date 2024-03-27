import mysql.connector

class events_length_static:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root',
                                            db='paper', charset='utf8')
        self.cursor = self.conn.cursor()

    def get_sents(self):
        sql = "select sent from events"
        self.cursor.execute(sql)
        sents = self.cursor.fetchall()
        self.cursor.close()
        self.conn.close()
        return sents

    def static_length(self):
        sents = self.get_sents()
        counts = [0, 0, 0]
        for sent in sents:
            length = len(sent[0])
            if 0 <= length < 15:
                counts[0] += 1
            elif 10 <= length < 30:
                counts[1] += 1
            else:
                counts[2] += 1
        return counts


print(events_length_static().static_length())

