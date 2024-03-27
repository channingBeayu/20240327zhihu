from .EventExtraction import EventExractor
from .EventTriples import TrunkExractor


class GetEvents:
    def __init__(self):
        self.eventExractor = EventExractor()
        self.trunkExractor = TrunkExractor()

    def get_events(self, doc):
        # 1、对doc抽取关系，涉及分句、抽取
        sent_infos = self.eventExractor.extract_doc_events(doc)
        # print(sent_infos)

        # # 2、对前半句和后半句分别提取主干
        # for sent_info in sent_infos:
        #     pre_part = sent_info['tuples']['pre_part']
        #     sent_info['pre_part'] = self.trunkExractor.event_to_trunk(pre_part)
        #     post_part = sent_info['tuples']['post_part']
        #     sent_info['post_part'] = self.trunkExractor.event_to_trunk(post_part)
        # print(sent_infos)

        return sent_infos


if __name__ == '__main__':
    doc = '环境很好，位置独立性很强，比较安静很切合店名，半闲居，偷得半日闲。点了比较经典的菜品，味道果然不错！烤乳鸽，超级赞赞赞，脆皮焦香，肉质细嫩，超好吃。艇仔粥料很足，香葱自己添加，很贴心。虽然金钱肚味道不错，但没有在广州吃的烂，然后牙口不好的慎点。凤爪很火候很好，推荐。最惊艳的是长寿菜，菜料十足，很新鲜，清淡又不乏味道，而且没有添加调料的味道，搭配的非常不错！'

    doc1 = '截至2008年9月18日12时，5·12汶川地震共造成69227人死亡，374643人受伤，17923人失踪，是中华人民共和国成立以来破坏力最大的地震，也是唐山大地震后伤亡最严重的一次地震。'

    sent_infos = GetEvents().get_events(doc)
    print(sent_infos)

    print(1)


