import gamera.core as gc
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images


class Syllable(object):

    letters_path = './letters/'
    letter_list = (
        'sa se si sp sr ss sti st su sy ca ci co cre cu'.split() +
        'ae be ca de do es et fe gr te us vo'.split() +
        'a b c d e f g h i j l m n o p q r s t u v x y'.split()
        )

    def __init__ (self, text):
        self.text = text
        self._resolve_text()
        #self._make_image()

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



if __name__ == "__main__":
    asdf = Syllable('domsyinisticasyebesetfegregrgrgr')
    print(asdf.resolved_text)
