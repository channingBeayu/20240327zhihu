import re
from pyltp import Segmentor
import os


class LtpParser():
    def __init__(self):
        LTP_DIR = r'F:\res\ltp_data'

        cws_model_path = os.path.join(LTP_DIR, 'cws.model')
        self.segmentor = Segmentor(cws_model_path)

    def doc_cut(self, doc):
        cutwords = list(self.segmentor.segment(doc))

        cuts = []
        for word in cutwords:
            cut = re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+', word, re.S)
            if cut:
                cuts.extend(cut)
        print(f'【{doc}】的分词结果：{cuts}')
        return cuts


if __name__ == '__main__':
    LtpParser().doc_cut('哈尔滨是一个城市')