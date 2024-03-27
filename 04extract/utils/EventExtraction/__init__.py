import re
from .SeqsEvent import EventsExtraction
from .CausalEvent import CausalityExractor


# 处理标点，取首句
def remove_headtail_punctuation(text, flag=True):
    pattern = re.compile(r'[,，\.。！!?？：；]')
    res = re.split(pattern, text)
    # flag为true说明是前句，取后面
    # flag为false说明是后句，取前面
    if not flag:
        res = res[::-1]
    for r in res:
        if r:
            print(r)
            return r
    return ''


class EventExractor():
    def __init__(self):
        self.eventsExtractor = EventsExtraction()
        self.causalExtractor = CausalityExractor()

    def extract_doc_events(self, doc):
        data = []
        data.extend(self.eventsExtractor.extract_main(doc))
        data.extend(self.causalExtractor.extract_main(doc))
        data = self.clean_data(data)
        return data

    def clean_data(self, data):
        res = []
        for item in data:
            # 1、前后句 处理标点
            pre_part = remove_headtail_punctuation(item['tuples']['pre_part'])
            post_part = remove_headtail_punctuation(item['tuples']['post_part'], False)

            # 判断长度是否大于5
            if len(pre_part) < 5 or len(post_part) < 5:
                continue
            item['tuples']['pre_part'] = pre_part
            item['tuples']['post_part'] = post_part

            res.append(item)

        return res



def t1():
    extractor = EventsExtraction()
    datas = extractor.extract_main('虽然你做了坏事，但我觉得你是好人。一旦时机成熟，就坚决推行')
    print(datas)
    print(extractor.stats(datas))


def t2():
    content = """
    因为11，所以22.
    """
    content1 = """
    截至2008年9月18日12时，5·12汶川地震共造成69227人死亡，374643人受伤，17923人失踪，是中华人民共和国成立以来破坏力最大的地震，也是唐山大地震后伤亡最严重的一次地震。
    """
    content2 = '''
    2015年1月4日下午3时39分左右，贵州省遵义市习水县二郎乡遵赤高速二郎乡往仁怀市方向路段发生山体滑坡，发生规模约10万立方米,导致多辆车被埋，造成交通双向中断。此事故引起贵州省委、省政府的高度重视，省长陈敏尔作出指示，要求迅速组织开展救援工作，千方百计实施救援，减少人员伤亡和财物损失。遵义市立即启动应急救援预案，市应急办、公安、交通、卫生等救援力量赶赴现场救援。目前，灾害已造成3人遇难1人受伤，一辆轿车被埋。
    当地时间2010年1月12日16时53分，加勒比岛国海地发生里氏7.3级大地震。震中距首都太子港仅16公里，这个国家的心脏几成一片废墟，25万人在这场骇人的灾难中丧生。此次地震中的遇难者有联合国驻海地维和部队人员，其中包括8名中国维和人员。虽然国际社会在灾后纷纷向海地提供援助，但由于尸体处理不当导致饮用水源受到污染，灾民喝了受污染的水后引发霍乱，已致至少2500多人死亡。
    '''
    extractor = CausalityExractor()
    datas = extractor.extract_main(content2)
    print(datas)

    # datas = extractor.extract_main(content2)
    # for data in datas:
    #     print('******'*4)
    #     print('cause', ''.join([word.split('/')[0] for word in data['cause'].split(' ') if word.split('/')[0]]))
    #     print('tag', data['tag'])
    #     print('effect', ''.join([word.split('/')[0] for word in data['effect'].split(' ') if word.split('/')[0]]))


def t3():
    doc = '''
        2015年1月4日下午3时39分左右，贵州省遵义市习水县二郎乡遵赤高速二郎乡往仁怀市方向路段发生山体滑坡，发生规模约10万立方米,导致多辆车被埋，造成交通双向中断。此事故引起贵州省委、省政府的高度重视，省长陈敏尔作出指示，要求迅速组织开展救援工作，千方百计实施救援，减少人员伤亡和财物损失。遵义市立即启动应急救援预案，市应急办、公安、交通、卫生等救援力量赶赴现场救援。目前，灾害已造成3人遇难1人受伤，一辆轿车被埋。
        当地时间2010年1月12日16时53分，加勒比岛国海地发生里氏7.3级大地震。震中距首都太子港仅16公里，这个国家的心脏几成一片废墟，25万人在这场骇人的灾难中丧生。此次地震中的遇难者有联合国驻海地维和部队人员，其中包括8名中国维和人员。虽然国际社会在灾后纷纷向海地提供援助，但由于尸体处理不当导致饮用水源受到污染，灾民喝了受污染的水后引发霍乱，已致至少2500多人死亡。
        '''
    exractor = EventExractor()
    data = exractor.extract_doc_events(doc)
    print(data)


# t3()


