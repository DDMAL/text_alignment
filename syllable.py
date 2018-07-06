import gamera.core as gc
import matplotlib.pyplot as plt
import itertools
import fastdtw
from gamera.plugins.image_utilities import union_images

class Syllable(object):

    gap_ignore = 10
    line_step = 2

    letters_path = './letters/'
    letter_list = (
        'sa se si sp sr ss sti st su sy ca ci co cre cu'.split() +
        'ae be ca de do es et fe gr te us vo'.split() +
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

        self._extract_features()

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

        self.image = result

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
        sqs = {}
        res['volume'] = self.image.volume()[0]
        res['black_area'] = self.image.black_area()[0]

        skeleton_feats = self.image.skeleton_features()
        for i,f in enumerate(skeleton_feats):
            res['skeleton_' + str(i)] = f

        sqs['horizontal_gaps'] = [self._runs_line(x, False) for x in range(0, self.image.nrows, Syllable.line_step)]
        sqs['vertical_gaps'] = [self._runs_line(x, True) for x in range(0, self.image.ncols, Syllable.line_step)]
        sqs['volume64regions'] = self.image.volume64regions()

        self.sequence_features = sqs
        self.features = res

if __name__ == "__main__":
    gc.init_gamera()
    asdf = Syllable('domssvo')
    print(asdf.resolved_text)
