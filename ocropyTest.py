import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import textAlignPreprocessing as preproc
import os
import PIL
import numpy as np
import textSeqCompare as tsc
import subprocess
from PIL import Image, ImageDraw, ImageFont
reload(preproc)
reload(tsc)

filename = 'salzinnes_31'
ocropus_model = './ocropy-master/models/salzinnes_model-00054500.pyrnn.gz'
parallel = 2
median_line_mult = 2

# removes some special characters from OCR output. ideally these would be useful but not clear how
# best to integrate them into the alignment algorithm. unidecode doesn't seem to work with these
# either
def clean_special_chars(inp):
    inp = inp.replace('~', '')
    inp = inp.replace('\xc4\x81', 'a')
    inp = inp.replace('\xc4\x93', 'e')
    # there is no i with bar above in unicode (???)
    inp = inp.replace('\xc5\x8d', 'o')
    inp = inp.replace('\xc5\xab', 'u')
    return inp


# get raw image of text layer and preform preprocessing to find text lines
raw_image = gc.load_image('./png/' + filename + '_text.png')
image, staff_image = preproc.preprocess_images(raw_image, None)
cc_lines, lines_peak_locs = preproc.identify_text_lines(image)

# get bounding box around each line, with padding (does padding affect ocropus output?)
cc_strips = []
for line in cc_lines:
    pad = 0
    x_min = min(c.offset_x for c in line) - pad
    x_max = max(c.offset_x + c.width for c in line) + pad
    y_max = max(c.offset_y + c.height for c in line) + pad

    # we want to cut off the tops of large capital letters, because that's how the model was
    # trained. set the top to be related to the median rather than the minimum y-coordinate
    y_min = min(c.offset_y for c in line)
    y_med_height = np.median([c.height for c in line]) * median_line_mult
    y_min = max(y_max - y_med_height, y_min)

    cc_strips.append(image.subimage((x_min, y_min), (x_max, y_max)))

# make directory to do stuff in
dir = 'wkdir_' + filename
if not os.path.exists(dir):
    subprocess.check_call("mkdir " + dir, shell=True)

# save strips to directory
for i, strip in enumerate(cc_strips):
    strip.save_image('./{}/{}_{}.png'.format(dir, filename, i))

# call ocropus command to do OCR on each saved line strip
ocropus_command = 'ocropus-rpred -Q {} --nocheck --llocs -m {} \'{}/*.png\''.format(parallel, ocropus_model, dir)
subprocess.check_call(ocropus_command, shell=True)

# read character position results from llocs file
all_chars = []
other_chars = []
for i in range(len(cc_strips)):
    locs_file = './{}/{}_{}.llocs'.format(dir, filename, i)
    with open(locs_file) as f:
        locs = [line.rstrip('\n') for line in f]

    x_min = cc_strips[i].offset_x
    y_min = cc_strips[i].offset_y

    text_line = []
    for l in locs:
        lsp = l.split('\t')

        if lsp[0] == '~' or lsp[0] == '':
            other_chars.append((clean_special_chars(lsp[0]), float(lsp[1]) + x_min, y_min))
            continue
        all_chars.append((clean_special_chars(lsp[0]), float(lsp[1]) + x_min, y_min))

# delete working directory
# subprocess.check_call("rm -r " + dir, shell=True)

# get full ocr transcript
ocr = ''.join(x[0] for x in all_chars)

transcript = tsc.read_file('./png/' + filename + '_transcript.txt')
tra_align, ocr_align = tsc.process(transcript, ocr)

# terrible hacky fix:
tra_align = tra_align[1:] + tra_align[0]
ocr_align = ocr_align[1:] + ocr_align[0]

align_transcript_chars = []

# insert gaps into ocr output based on alignment string
for i, char in enumerate(ocr_align):
    if char == '_':
        all_chars.insert(i, ('_', 0, 0))

for i, ocr_char in enumerate(all_chars):
    tra_char = tra_align[i]

    if not (tra_char == '_' or ocr_char[0] == '_'):
        align_transcript_chars.append([tra_char, ocr_char[1], ocr_char[2]])
    elif (tra_char != '_'):
        align_transcript_chars[-1][0] += tra_char


# # DRAW ON ORIGINAL IMAGE
im = image.to_greyscale().to_pil()
text_size = 80
fnt = ImageFont.truetype('Arial.ttf', text_size)
draw = ImageDraw.Draw(im)
# for i, line in enumerate(all_chars_lines):
#
#     x_min = cc_strips[i].offset_x
#     y_min = cc_strips[i].offset_y
#     # draw.rectangle((char[1], char[2]), outline=0)
#     for char in line:
#         draw.text((x_min + int(char[1]), y_min - text_size), char[0], font=fnt, fill=0)

for i, char in enumerate(align_transcript_chars):
    draw.text((char[1], char[2] - text_size), char[0], font=fnt, fill='gray')
    draw.line([char[1], char[2], char[1], char[2] + 100], fill='gray', width=10)

im.save('testimg.png')
im.show()
