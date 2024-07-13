import re


def get_link_weight(link_label):
    type = re.findall(r'\[(.*?)\]', link_label)[0]
    if type == 'but':
        return 0.3
    elif type == 'causal':
        return 2.0
    elif type == 'condition':
        return 1.5
    elif type == 'more':
        return 1.0
    elif type == 'seq':
        return 2.0