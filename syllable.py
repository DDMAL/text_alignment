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

    # 'syl_groups' references the list of syllables for this element (0 to max_syls_per_element)
    # 'cc_groups' is the list of ccs that comprise this syllable (0 to forward_branches)
    # 'cost' is the cost associated with taking these ccs to estimate these syllables
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


def get_cost_of_element(cc_group, syl_group, spaces, median_area, median_space, scale=2):
    # MAXIMISE SPACES AT THE ENDS OF WORDS
    # MINIMISE INTERNAL SPACES
    # THAT IS ALL

    median_area *= scale
    # median_space *= scale

    cc_area = sum(min(x.black_area()[0], median_area) for x in cc_group)
    syl_area = sum(x.area for x in syl_group)
    area_diff = (abs(cc_area - syl_area))

    # largest_internal_space = 0 if len(spaces) == 1 else max(spaces[:-1])
    internal_space = sum(x for x in spaces[:-1] if x > median_space)

    cost = internal_space + area_diff

    if syl_group and (syl_group[-1].word_end) and spaces[-1] < median_space:
        cost += np.inf

    cost = max(0, cost)

    return round(cost, 3)


def make_align_seq_from_path(path, cc_lines_flat, syllables, spaces):
    cc_groups = []
    syl_groups = []
    costs = []

    median_area = np.median([x.black_area()[0] for x in cc_lines_flat])
    median_space = np.median(spaces)

    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i+1]

        cc_groups.append(cc_lines_flat[start_node[0]:end_node[0]])
        sps = spaces[start_node[0]:end_node[0]]
        syl_groups.append(syllables[start_node[1]:end_node[1]])
        costs.append(get_cost_of_element(cc_groups[-1], syl_groups[-1], sps, median_area, median_space))

    return AlignSequence(cc_groups=cc_groups, syl_groups=syl_groups, costs=costs)
