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


def recognize_text_strips(img, cc_strips, path_to_ocr_model, verbose=False):

    predictor = Predictor(checkpoint=path_to_ocr_model)

    # x, y, width, height
    strips = []
    for ls in cc_strips:
        x, y, w, h = ls
        # WHY IS Y FIRST? WHAT'S THE DEAL
        strip = img[y:y + h, x:x + w]
        strips.append(strip)

    results = []
    for r in predictor.predict_raw(strips, progress_bar=verbose):
        results.append(r)

    all_chars = []

    # iterate over results and make charbox objects out of every character
    for i, cs in enumerate(cc_strips):

        strip_x_min, strip_y_min, strip_width, strip_height = cs
        res_line = [
            CharBox(
                clean_special_chars(x.chars[0].char),
                (x.global_start, strip_y_min),
                (x.global_end, strip_y_min + strip_height))
            for x in results[i].prediction.positions
            ]

        all_chars += res_line

    # remove all chars that are empty or otherwise invalid
    all_chars = list(filter(lambda x: x.char not in ['', '~'], all_chars))
    return all_chars


if __name__ == '__main__':

    import textAlignPreprocessing as preproc

    fname = 'salzinnes_378'
    raw_image = cv.imread('./png/{}_text.png'.format(fname))

    img_bin, img_eroded, angle = preproc.preprocess_images(raw_image)
    line_strips, lines_peak_locs, proj = preproc.identify_text_lines(img_eroded)

    path_to_ocr_model = './models/mcgill_salzinnes/1.ckpt'

    all_chars = recognize_text_strips(img_bin, line_strips, path_to_ocr_model, True)
