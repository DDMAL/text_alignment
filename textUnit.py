import gamera.core as gc
import matplotlib.pyplot as plt
import itertools

import numpy as np

from gamera.plugins.image_utilities import union_images


class textUnit(object):

    gap_ignore = 10
    line_step = 4

    letters_path = './letters/'
    letter_list = (
        'cae cre est rex sti tet tri'.split() +
        'ae am an be bo ca ch ci co cu de do ec em en es et fa fe fi fo gr pe po om on ra sa se si sp sr ss st su sy ta te ti tu us um un vo xs'.split() +
        'a b c d e f g h i j l m n o p q r s t u v x y ae_2 r_2 e_2 d_2 v_2 m_2 b_2'.split() +
        '_wildcard_unit'
        )

    prototypes = {}

    # necessary to use a helper function to load images into a dictionary comprehension because
    # python does strange things to scope in the expression part of a comprehension (????????)
    def _dict_helper(A, B):
        return {x: gc.load_image(A + x + '.png').to_onebit() for x in B}

    chunk_images = _dict_helper(letters_path, letter_list)

    def __init__(self, image=None, text=None, is_wildcard=None):

        if not (bool(text) ^ bool(image)):
            raise ValueError('Supply an image or text, but not both')

        self.is_wildcard = is_wildcard

        if text:
            self.is_synthetic = True
            self.text = text
            self.image = textUnit.chunk_images[text].trim_image()

        if image:
            self.is_synthetic = False
            self.image = image
            self.text = None

        self.box_index = None
        self._extract_features()

        self.ul = gc.Point(self.image.offset_x, self.image.offset_y)
        self.lr = gc.Point(
            self.image.offset_x + self.image.ncols,
            self.image.offset_y + self.image.nrows
            )

        self.nrows = self.image.nrows
        self.ncols = self.image.ncols
        self.offset_x = self.image.offset_x
        self.offset_y = self.image.offset_y

    def _index_in_seq(self, subseq, seq):
        i, n, m = -1, len(seq), len(subseq)
        try:
            while True:
                i = seq.index(subseq[0], i + 1, n - m + 1)
                if subseq == seq[i:i + m]:
                    return i
        except ValueError:
            return -1

    def _runs_line(self, ind, vertical=False, gap_ignore=gap_ignore, line_step=line_step):

        if vertical:
            cross_line = [self.image[ind, x] for x in range(self.image.nrows)]
        else:
            cross_line = [self.image[x, ind] for x in range(self.image.ncols)]

        switch_points = [i for i, x in enumerate(cross_line) if cross_line[i-1] != x]

        for i in reversed(range(len(switch_points)-2)):

            gap_size = switch_points[i+1] - switch_points[i]
            if gap_size >= gap_ignore:
                continue
            del switch_points[i+1]

        return len(switch_points)

    def _extract_features(self):
        res = {}

        res['volume'] = self.image.volume()[0]
        res['black_area'] = self.image.black_area()[0]
        res['aspect_ratio'] = self.image.aspect_ratio()[0]
        res['diagonal_projection'] = self.image.diagonal_projection()[0]

        skeleton_feats = self.image.skeleton_features()
        for i, f in enumerate(skeleton_feats):
            res['skeleton_' + str(i)] = f

        volume_feats = self.image.volume64regions()
        for i, f in enumerate(volume_feats):
            res['volume_region_' + str(i)] = f

        moments_feats = self.image.moments()
        for i, f in enumerate(moments_feats):
            res['moments_' + str(i)] = f

        self.features = res


def knn_search(train_set, test_syl, k=5):
    '''
    given a list of textUnits as a train set and a single textUnit to search for, returns the
    nearest textUnit to the test textUnit in the feature space.
    '''

    feature_keys = train_set[0].features.keys()
    closest = [(None, np.inf)]
    max_val = np.inf

    for train_syl in train_set:
        dist = 0

        for fk in feature_keys:
            df = test_syl.features[fk] - train_syl.features[fk]
            dist += df * df

            # break out early if possible
            if dist > max_val:
                break

        if dist > max_val:
            continue

        # if we didn't break out early, this is a candidate for entry
        # key with max value
        if len(closest) >= k:
            max_key = max(closest, key=lambda x: x[1])
            closest.remove(max_key)

        closest.append((train_syl, dist))
        max_val = max(closest, key=lambda x: x[1])[1]

    closest.sort(key=lambda x: x[1])
    return closest


def get_prototypes():
    res = {}

    for l in textUnit.letter_list:
        res[l] = textUnit(text=l)

    res['*'] = textUnit(is_wildcard=True, text='_wildcard_unit')

    return res


def compare_units(a, b):
    feature_keys = a.features.keys()

    if a.is_wildcard or b.is_wildcard:
        return 0

    sum = 0

    for fk in feature_keys:
        sum += (a.features[fk] - b.features[fk]) * (a.features[fk] - b.features[fk])

    return sum


class unitSequence(object):
    def __init__(self, seq=None, skip_edges=None, used_edges=None,
            char_index=None, cost_arr=None, predicted_string=None):
        self.cost_arr = cost_arr if cost_arr else []
        self.seq = seq if seq else []
        self.skip_edges = skip_edges if skip_edges else []
        self.used_edges = used_edges if used_edges else []
        self.char_index = char_index if char_index else 0
        self.predicted_string = predicted_string if predicted_string else []

    def head(self):
        return self.seq[-1]

    def cost(self, moving_avg=False):
        if not moving_avg:
            return np.mean(self.cost_arr)

        res = []
        for i in range(len(self.cost_arr)):
            res.append(np.mean(self.cost_arr[0:i]))

        return res

    def equivalent(self):
        return [self.char_index, self.seq[-1][0], self.seq[-1][1]]

    def __repr__(self):
        return 'cost: {3}, index: {0.char_index}, len: {1}, ' \
            'base:{2}'.format(self, len(self.used_edges), self.seq[0], self.cost())


if __name__ == "__main__":
    gc.init_gamera()
    asdf = get_prototypes()
