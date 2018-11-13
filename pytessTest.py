import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import textAlignPreprocessing as preproc
import pytesseract
import PIL
import textSeqCompare as tsc
from PIL import Image, ImageDraw, ImageFont
reload(preproc)
reload(tsc)


def disp(image):
    image.to_greyscale().to_pil().show()
    return

filename = 'einsiedeln_001v'

raw_image = gc.load_image('./png/' + filename + '_text.png')
image, staff_image = preproc.preprocess_images(raw_image, None)
cc_lines, lines_peak_locs = preproc.identify_text_lines(image)

# get bounding box around each line, with padding
# cc_strips = [union_images(line) for line in cc_lines]
cc_strips = []
for line in cc_lines:
    pad = 10
    x_min = min(c.offset_x for c in line) - pad
    y_min = min(c.offset_y for c in line) - pad
    x_max = max(c.offset_x + c.width for c in line) + pad
    y_max = max(c.offset_y + c.height for c in line) + pad
    cc_strips.append(image.subimage((x_min, y_min), (x_max, y_max)))

chars = []
strings = []

conf_str = '-c tessedit_char_whitelist=.abcdefghilmnopqrstuvy ' \
           'textord_space_size_is_variable=1 language_model_ngram_on=0 load_system_dawg=0' \
           'load_freq_dawg=0 -psm 7'

for strip in cc_strips:
    res = pytesseract.image_to_boxes(strip.to_greyscale().to_pil(), config=conf_str, lang='lat')
    strings.append(pytesseract.image_to_string(strip.to_greyscale().to_pil(), config=conf_str, lang='lat'))
    lines = str(res).split('\n')
    for line in lines:
        res = line.split(' ')
        ul = (strip.ul.x + int(res[1]), strip.ul.y + int(res[2]))
        lr = (strip.ul.x + int(res[3]), strip.ul.y + int(res[4]))
        chars += [(res[0], ul, lr)]

# ocr = str(' '.join(strings))
# ocr = ocr.replace('\n', '')
# transcript = tsc.read_file('./png/' + filename + '_transcript.txt')
#
# tra_align, ocr_align = tsc.process(transcript, ocr)


# DRAW ON ORIGINAL IMAGE
im = image.to_greyscale().to_pil()
text_size = 40
fnt = ImageFont.truetype('Arial.ttf', text_size)
draw = ImageDraw.Draw(im)
for char in chars:
    draw.rectangle((char[1], char[2]), outline=0)
    draw.text((char[1][0], char[1][1] - text_size), char[0], font=fnt, fill=0)
im.save('testimg.png')
im.show()
