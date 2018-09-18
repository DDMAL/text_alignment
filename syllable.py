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

    def __init__(self, positions):
        self.score = 0
        self.positions = positions
        self.completed = False

    def __repr__(self):
        return str(self.__dict__)

    def head(self):
        if len(self.positions):
            return self.positions[-1]
        else:
            return 0

    def equivalence(self):
        if self.completed:
            return True
        else:
            return (len(self.positions), self.head())
