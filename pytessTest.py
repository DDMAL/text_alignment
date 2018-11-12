import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import textAlignPreprocessing as preproc
import pytesseract
import PIL
from PIL import Image
reload(preproc)

filename = 'salzinnes_25'

raw_image = gc.load_image('./png/' + filename + '_text.png')
image, staff_image = preproc.preprocess_images(raw_image, None)
cc_lines, lines_peak_locs = preproc.identify_text_lines(image)
cc_strips = [union_images(line) for line in cc_lines]

raw_strips = []
for s in cc_strips:
    raw_strips.append(raw_image.subimage(s.ul, s.lr))


strings = []

conf_str = '-c tessedit_char_whitelist=.abcdefghilmnopqrstuvyzABCDEFGHIJLMNOPQRSTUVYZ ' \
           'textord_space_size_is_variable=1 language_model_ngram_on=0 -psm 7'

for x in cc_strips:
    strings.append(pytesseract.image_to_string(x.to_greyscale().to_pil(), config=conf_str, lang='deu_frak'))
    print(strings[-1])
