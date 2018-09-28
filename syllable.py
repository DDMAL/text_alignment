import numpy as np

class Syllable(object):

    def __init__(self, text=None, width=None, word_begin=None, word_end=None, area=None):
        self.text = text
        self.width = width
        self.word_begin = word_begin
        self.word_end = word_end
        self.area = area

    def __repr__(self):
        return str(self.__dict__)


class AlignSequence(object):

    def __init__(self, syl_groups=[], cc_groups=[], costs=[]):

        if len(syl_groups) != len(cc_groups):
            raise ValueError('syl_groups must be the same length as cc_groups')

        self.syl_groups = syl_groups
        self.cc_groups = cc_groups
        self.costs = costs
        self.completed = False

    def last_syl_index(self):
        return sum([len(x) for x in self.syl_groups]) - 1

    def last_cc_index(self):
        return sum([len(x) for x in self.cc_groups]) - 1

    def num_elements(self):
        return len(self.cc_groups)

    def __repr__(self):
        s = 'size: {} position: ({}, {}) cost: {}'.format(self.num_elements(), self.last_cc_index(), self.last_syl_index(), str(self.costs))
        return s

def get_cost_of_element(cc_group, syl_group, spaces, spaces_factor = 2):

    cc_area = sum(x.black_area()[0] for x in cc_group)
    syl_area = sum(x.area for x in syl_group)
    area_diff = abs(cc_area - syl_area)

    cc_width = max(cc.lr.x for cc in cc_group) - min(cc.ul.x for cc in cc_group)
    # cc_width = sum(x.ncols for x in cc_group)
    syl_width = sum(x.width for x in syl_group)


    largest_internal_space = 0 if len(spaces) == 1 else max(spaces[:-1])
    internal_space = 0 if len(spaces) == 1 else sum(spaces[:-1])
    cc_width -= internal_space
    width_diff = abs(cc_width - syl_width) ** 2

    cost = width_diff + largest_internal_space ** 2

    min_space = np.median(spaces) * spaces_factor
    if largest_internal_space > min_space:
        cost = np.inf

    if syl_group and syl_group[-1].word_end:

        if spaces[-1] < min_space:
            cost = np.inf
        # else:
            # cost = max(0, cost - spaces[-1])

    return round(cost, 3)

def make_align_seq_from_path(path, cc_lines_flat, syllables, spaces):
    cc_groups = []
    syl_groups = []
    costs = []

    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i+1]

        cc_groups.append(cc_lines_flat[start_node[0]:end_node[0]])
        sps = spaces[start_node[0]:end_node[0]]
        syl_groups.append(syllables[start_node[1]:end_node[1]])
        costs.append(get_cost_of_element(cc_groups[-1], syl_groups[-1], sps))

    return AlignSequence(cc_groups=cc_groups,syl_groups=syl_groups,costs=costs)
