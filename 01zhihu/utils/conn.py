import mysql.connector


class Conn:
    def __init__(self):
        self.conn = mysql.connector.connect(host='localhost', user='root', passwd='root',
                                            db='zhihu', charset='utf8')
        self.cursor = self.conn.cursor()

    def get_urls(self):
        sql = "select url from 01docs_info"
        self.cursor.execute(sql)
        urls = self.cursor.fetchall()
        return urls


    def save_df(self, table_name, df):
        column_names = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

        for row in df.itertuples(index=False):  # 遍历DataFrame中的每一行
            print("正在插入数据:", row)  # 输出正在插入的数据
            self.cursor.execute(insert_query, row)  # 执行SQL插入语句

        self.conn.commit()

    def get_contents(self):
        sql = "select doc_id, content from 01docs_info"
        self.cursor.execute(sql)
        doc_infos = self.cursor.fetchall()
        return doc_infos

    def close(self):
        self.cursor.close()
        self.conn.close()