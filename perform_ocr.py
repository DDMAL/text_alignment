from calamari_ocr.ocr.predictor import Predictor
import cv2 as cv


class CharBox(object):
    __slots__ = ['char', 'ul', 'lr', 'ulx', 'lrx', 'uly', 'lry', 'width', 'height']

    def __init__(self, char, ul=None, lr=None):

        self.char = char
        if (ul is None) or (lr is None):
            self.ul = None
            self.lr = None
            return
        self.ul = tuple(ul)
        self.lr = tuple(lr)
        self.ulx = ul[0]
        self.lrx = lr[0]
        self.uly = ul[1]
        self.lry = lr[1]
        self.width = lr[0] - ul[0]
        self.height = lr[1] - ul[1]

    def __repr__(self):
        if self.ul and self.lr:
            return 'CharBox: \'{}\' at {}, {}'.format(self.char, self.ul, self.lr)
        else:
            return 'CharBox: \'{}\' empty'.format(self.char)


def clean_special_chars(inp):
    '''
    removes some special characters from OCR output. ideally these would be useful but not clear how
    best to integrate them into the alignment algorithm. unidecode doesn't seem like these either
    '''
    inp = inp.replace('~', '')
    # inp = inp.replace('\xc4\x81', 'a')
    # inp = inp.replace('\xc4\x93', 'e')
    # # there is no i with bar above in unicode (???)
    # inp = inp.replace('\xc5\x8d', 'o')
    # inp = inp.replace('\xc5\xab', 'u')
    return inp


def recognize_text_strips(img, cc_strips, path_to_ocr_model, verbose=True):

    predictor = Predictor(checkpoint='path_to_ocr_model)

    # x, y, width, height
    strips = []
    for ls in cc_strips:
        x, y, w, h = ls
        # WHY IS Y FIRST? WHAT'S THE DEAL
        strip = img[y:y + h, x:x + w]
        strips.append(strip)

    results = []
    for r in predictor.predict_raw(strips):
        results.append(r)

    

    # read character position results from llocs file
    all_chars = []
    other_chars = []
    for i in range(len(cc_strips)):
        locs_file = './{}/_{}.llocs'.format(wkdir_name, i)
        with io.open(locs_file, encoding='utf-8') as f:
            locs = [line.rstrip('\n') for line in f]

        x_min = cc_strips[i].offset_x
        y_min = cc_strips[i].offset_y
        y_max = cc_strips[i].offset_y + cc_strips[i].height

        # note: ocropus seems to associate every character with its RIGHTMOST edge. we want the
        # left-most edge, so we associate each character with the previous char's right edge
        text_line = []
        prev_xpos = x_min
        for l in locs:
            lsp = l.split('\t')
            cur_xpos = int(np.round(float(lsp[1]) + x_min))

            ul = (prev_xpos, y_min)
            lr = (cur_xpos, y_max)

            if lsp[0] == '~' or lsp[0] == '':
                new_box = CharBox(unicode(lsp[0]), ul, lr)
                other_chars.append(new_box)
            else:
                new_box = CharBox(clean_special_chars(lsp[0]), ul, lr)
                all_chars.append(new_box)

            prev_xpos = cur_xpos

    return all_chars


if __name__ == '__main__':

    import textAlignPreprocessing as preproc

    fname = 'salzinnes_378'
    raw_image = cv.imread('./png/{}_text.png'.format(fname))

    img_bin, img_eroded, angle = preproc.preprocess_images(raw_image)
    line_strips, lines_peak_locs, proj = preproc.identify_text_lines(img_eroded)

    predictor = Predictor(checkpoint='./models/mcgill_salzinnes/1.ckpt')

    # x, y, width, height
    strips = []
    for ls in line_strips:
        x, y, w, h = ls
        # WHY IS Y FIRST? WHAT'S THE DEAL WITH THIS
        strip = img_eroded[y:y + h, x:x + w]
        strips.append(strip)

    results = []

    for r in predictor.predict_raw(strips):
        results.append(r)
