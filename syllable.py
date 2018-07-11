import gamera.core as gc
import matplotlib.pyplot as plt
import itertools
from fastdtw import fastdtw
import numpy as np

from gamera.plugins.image_utilities import union_images

class Syllable(object):

    gap_ignore = 10
    line_step = 4

    letters_path = './letters/'
    letter_list = (
        'cae cre est rex nus sti tri'.split() +
        'ae be bo ca ch ci co cu de do ec es et fa fe fi fo gr pe po ra sa se si sp sr ss st su sy ta te ti tu us vo'.split() +
        'a b c d e f g h i j l m n o p q r s t u v x y'.split()
        )

    #necessary to use a helper function to load images into a dictionary comprehension because
    #python does strange things to scope in the expression part of a comprehension (????????)
    def _dict_helper(A,B):
        return {x:gc.load_image(A + x + '.png').to_onebit() for x in B}

    chunk_images = _dict_helper(letters_path,letter_list)

    def __init__ (self, text = None, image = None):

        if not (bool(text) ^ bool(image)):
            raise ValueError('Supply an image or text, but not both')

        if text:
            self.is_synthetic = True
            self.text = text
            self._resolve_text()
            self._make_image()

        if image:
            self.is_synthetic = False
            self.image = image
            self.text = None

        self.box_index = None
        self._extract_features()

        self.ul = gc.Point(self.image.offset_x,self.image.offset_y)
        self.lr = gc.Point(
            self.image.offset_x + self.image.ncols,
            self.image.offset_y + self.image.nrows
            )

        self.left = self.image.offset_x
        self.up = self.image.offset_y
        self.right = self.image.offset_x + self.image.ncols
        self.down = self.image.offset_y + self.image.nrows

    def _index_in_seq(self, subseq, seq):
        i, n, m = -1, len(seq), len(subseq)
        try:
            while True:
                i = seq.index(subseq[0], i + 1, n - m + 1)
                if subseq == seq[i:i + m]:
                   return i
        except ValueError:
            return -1

    def _resolve_text(self):
        result = list(self.text)

        for chunk in self.letter_list:
            l = list(chunk)
            index = self._index_in_seq(l,result)

            while index != -1:
                result = result[:index] + result[index + len(chunk):]
                result.insert(index, "@" + chunk)
                index = self._index_in_seq(l,result)

        result[:] = [r.replace('@','') for r in result]

        self.resolved_text = result

    def _make_image(self):
        #given a list of individual chunk images, stitch them together into an image of a syllable.
        #
        ims = [Syllable.chunk_images[x] for x in self.resolved_text]

        padding = 1
        height = max(x.nrows for x in ims)
        width = sum(x.ncols for x in ims) + (padding * (len(ims) - 1))
        result = gc.Image(gc.Point(0,0),gc.Point(width,height))

        cur_left_bound = 0

        for img in ims:

            for coord in itertools.product(range(img.ncols),range(img.nrows)):
                trans_down = result.nrows - img.nrows
                new_coord = (coord[0] + cur_left_bound, coord[1] + trans_down)
                result.set(new_coord,img.get(coord))

            cur_left_bound += padding + img.ncols


        self.image = result.trim_image()

    def _runs_line(self, ind, vertical = False, gap_ignore = gap_ignore, line_step = line_step):

        if vertical:
            cross_line = [self.image[ind,x] for x in range(self.image.nrows)]
        else:
            cross_line = [self.image[x,ind] for x in range(self.image.ncols)]

        switch_points = [i for i, x in enumerate(cross_line) if cross_line[i-1] != x]

        for i in reversed(range(len(switch_points)-2)):

            gap_size = switch_points[i+1] - switch_points[i]
            if gap_size >= gap_ignore:
                continue
            del switch_points[i+1]

        return len(switch_points)

    def _extract_features(self):
        res  = {}

        res['volume'] = self.image.volume()[0]
        res['black_area'] = self.image.black_area()[0]
        res['area'] = self.image.area()[0]
        res['diagonal_projection'] = self.image.diagonal_projection()[0]

        skeleton_feats = self.image.skeleton_features()
        for i,f in enumerate(skeleton_feats):
            res['skeleton_' + str(i)] = f

        volume_feats = self.image.volume64regions()
        for i,f in enumerate(volume_feats):
            res['volume_region_' + str(i)] = f

        moments_feats = self.image.moments()
        for i,f in enumerate(moments_feats):
            res['moments_' + str(i)] = f

        self.features = res

# def compare(A, B):
#     dist = []
#
#     seqs_a = A.sequence_features.values()
#     seqs_b = B.sequence_features.values()
#
#     for n in range(len(seqs_a)):
#         a_pts = [[i,x] for i,x in enumerate(seqs_a[n])]
#         b_pts = [[i,x] for i,x in enumerate(seqs_b[n])]
#         dist.append(fastdtw(a_pts, b_pts)[0])
#
#     return dist

def knn_search(train_set, test_syl, k = 5):
    '''
    given a list of syllables as a train set and a single syllable to search for, returns the
    nearest syllable to the test syllable in the feature space.
    '''

    feature_keys = train_set[0].features.keys()
    closest = [(None, np.inf)]
    max_val = np.inf

    for train_syl in train_set:
        dist = 0

        for fk in feature_keys:
            df = test_syl.features[fk] - train_syl.features[fk]
            dist += df * df

            #break out early if possible
            if dist > max_val:
                break

        if dist > max_val:
            continue

        #if we didn't break out early, this is a candidate for entry
        #key with max value
        if len(closest) >= k:
            max_key = max(closest, key = lambda x: x[1])
            closest.remove(max_key)

        closest.append((train_syl,dist))
        max_val = max(closest, key = lambda x: x[1])[1]

    closest.sort(key = lambda x: x[1])
    return closest

if __name__ == "__main__":
    gc.init_gamera()
    asdf = Syllable('domssvo')
    hjkl = Syllable('asdssvo')
