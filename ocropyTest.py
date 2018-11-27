import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import textAlignPreprocessing as preproc
import os
import PIL
import textSeqCompare as tsc
import subprocess
from PIL import Image, ImageDraw, ImageFont
reload(preproc)
reload(tsc)

def disp(image):
    image.to_greyscale().to_pil().show()
    return


filename = 'einsiedeln_002v'
# ocropus_model = './ocropy-master/models/latin_salz_model_wip.gz'
ocropus_model = './ocropy-master/models/salzinnes_model-00003000.pyrnn.gz'


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

# make directory to do stuff in
dir = 'ocropus_' + filename

if not os.path.exists(dir):
    subprocess.check_call("mkdir " + dir, shell=True)

# save strips to directory
for i, strip in enumerate(cc_strips):
    strip.save_image('./{}/{}_{}.png'.format(dir, filename, i))

ocropus_command = 'ocropus-rpred -Q 2 --nocheck --llocs -m ' + ocropus_model + \
                    ' \'' + dir + '/*.png\''
subprocess.check_call(ocropus_command, shell=True)

all_chars_lines = []
for i in range(len(cc_strips)):
    locs_file = './{}/{}_{}.llocs'.format(dir, filename, i)
    with open(locs_file) as f:
        locs = [line.rstrip('\n') for line in f]

    text_line = []
    for l in locs:
        lsp = l.split('\t')
        text_line.append((lsp[0], float(lsp[1])))
    all_chars_lines.append(text_line)

subprocess.check_call("rm -r " + dir, shell=True)

# get full ocr transcript
ocr = ''.join([item[0] for sublist in all_chars_lines for item in sublist])
transcript = tsc.read_file('./png/' + filename + '_transcript.txt')
# transcript = transcript.replace(' ', '')
tra_align, ocr_align = tsc.process(transcript, ocr)

# # DRAW ON ORIGINAL IMAGE
# im = image.to_greyscale().to_pil()
# text_size = 40
# fnt = ImageFont.truetype('Arial.ttf', text_size)
# draw = ImageDraw.Draw(im)
# for char in chars:
#     draw.rectangle((char[1], char[2]), outline=0)
#     draw.text((char[1][0], char[1][1] - text_size), char[0], font=fnt, fill=0)
# im.save('testimg.png')
# im.show()
