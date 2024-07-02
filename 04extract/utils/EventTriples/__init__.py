from .triple_extraction import TripleExtractor


class TrunkExractor():
    def __init__(self):
        self.tripleExtractor = TripleExtractor()

    def event_to_trunk(self, event):
        svos = self.tripleExtractor.triples_main(event)
        # print('SVOs(主谓宾)', svo)
        if svos:
            return ''.join(svos[0])
        else:
            return event



if __name__ == '__main__':
    res = TrunkExractor().event_to_trunk('我感到非常荣幸')
    print('SVOs(主谓宾):', res)
