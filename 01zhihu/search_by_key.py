from utils.conn import Conn
from utils.get_urls import get_html
from utils.get_passage import GetPassages
from utils.get_answers import GetAnswers


class GetUrls():
    def __init__(self):
        self.conn = Conn()

    def main(self):
        urls_question, urls_passage = self.filter_urls(get_html('哈尔滨旅游感受'))
        questions_data = GetAnswers(urls_question).main()
        passages_data = GetPassages(urls_passage).main()
        self.conn.save_df('01docs_info', questions_data)
        self.conn.save_df('01docs_info', passages_data)
        self.conn.close()

    def filter_urls(self, urls):
        urls_question, urls_passage = urls[0], urls[1]
        saved_urls = self.conn.get_urls()

        filtered_urls_question = []
        for url in urls_question:
            if url not in saved_urls:
                filtered_urls_question.append(url)

        filtered_urls_passage = []
        for url in urls_passage:
            if url not in saved_urls:
                filtered_urls_passage.append(url)

        return filtered_urls_question, filtered_urls_passage


GetUrls().main()
