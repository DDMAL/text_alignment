class Syllable(object):

    def __init__(self, text=None, width=None, word_begin=None):
        self.text = text
        self.width = width
        self.word_begin = word_begin

    def __repr__(self):
        return str(self.__dict__)
