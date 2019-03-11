import gamera.core as gc
gc.init_gamera()
from gamera.plugins.image_utilities import union_images
import matplotlib.pyplot as plt
import textAlignPreprocessing as preproc
import os
import shutil
import numpy as np
import textSeqCompare as tsc
import latinSyllabification as latsyl
import subprocess

reload(preproc)
reload(tsc)
reload(latsyl)

ocropus_model = './salzinnes_model-00054500.pyrnn.gz'
parallel = 2
median_line_mult = 2

# there are some hacks to make this work on windows. no guarantees, and OCRopus will probably
# complain a lot. you will definitely have to use a model that is NOT zipped, or else go into the
# common.py file in ocrolib and change the way it's compressed from gunzip to gzip (gunzip is not
# natively available on windows). also, parallel processing does not work on windows.
on_windows = (os.name == 'nt')

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


def process(raw_image, transcript, wkdir_name='', parallel=parallel, median_line_mult=median_line_mult, ocropus_model=ocropus_model, verbose=False):

    #######################
    # -- PRE-PROCESSING --
    #######################

    # get raw image of text layer and preform preprocessing to find text lines
    # raw_image = gc.load_image('./png/' + filename + '_text.png')
    image, staff_image = preproc.preprocess_images(raw_image, None)
    cc_lines, lines_peak_locs = preproc.identify_text_lines(image)

    # get bounding box around each line, with padding
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
    dir = 'wkdir_' + wkdir_name
    if not os.path.exists(dir):
        subprocess.check_call("mkdir " + dir, shell=True)

    # save strips to directory
    for i, strip in enumerate(cc_strips):
        strip.save_image('./{}/{}_{}.png'.format(dir, wkdir_name, i))

    #################################
    # -- PERFORM OCR WITH OCROPUS --
    #################################

    # call ocropus command to do OCR on each saved line strip.
    if on_windows:
        cwd = os.getcwd()
        ocropus_command = 'python ./ocropy-master/ocropus-rpred ' \
            '--nocheck --llocs -m {} {}/{}/*'.format(ocropus_model, cwd, dir)
    else:
        # the presence of extra quotes \' around the path to be globbed makes a difference.
        # sometimes. it's unclear.
        ocropus_command = 'ocropus-rpred -Q {} ' \
            '--nocheck --llocs -m {} \'{}/*.png\''.format(parallel, ocropus_model, dir)

    print('running ocropus with: {}'.format(ocropus_command))
    subprocess.check_call(ocropus_command, shell=True)

    # read character position results from llocs file
    all_chars = []
    other_chars = []
    for i in range(len(cc_strips)):
        locs_file = './{}/{}_{}.llocs'.format(dir, wkdir_name, i)
        with open(locs_file) as f:
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
            cur_xpos = float(lsp[1]) + x_min

            ul = (prev_xpos, y_min)
            lr = (cur_xpos, y_max)

            if lsp[0] == '~' or lsp[0] == '':
                other_chars.append((lsp[0], ul, lr))
            else:
                all_chars.append((clean_special_chars(lsp[0]), ul, lr))

            prev_xpos = cur_xpos

    # delete working directory
    if on_windows:
        shutil.rmtree(dir)
    else:
        subprocess.check_call('rm -r ' + dir, shell=True)

    # get full ocr transcript
    ocr = ''.join(x[0] for x in all_chars)
    all_chars_copy = list(all_chars)

    ###################################
    # -- PERFORM AND PARSE ALIGNMENT --
    ###################################

    tra_align, ocr_align = tsc.perform_alignment(transcript, ocr, verbose=verbose)

    align_transcript_chars = []

    # insert gaps into ocr output based on alignment string. this causes all_chars to have gaps at the
    # same points as the ocr_align string does, and is thus the same length as tra_align.
    for i, char in enumerate(ocr_align):
        if char == '_':
            all_chars.insert(i, ('_', 0, 0))

    # this could very possibly go wrong (special chars, bug in alignment algorithm, etc) so better
    # make sure that this condition is holding at this point
    assert len(all_chars) == len(tra_align), 'all_chars not same length as alignment'

    for i, ocr_char in enumerate(all_chars):
        tra_char = tra_align[i]

        if not (tra_char == '_' or ocr_char[0] == '_'):
            align_transcript_chars.append([tra_char, ocr_char[1], ocr_char[2]])
        elif (tra_char != '_') and align_transcript_chars:
            align_transcript_chars[-1][0] += tra_char
        elif (tra_char != '_'):
            # a tricky case: what if the first letter of the transcript is assigned to a gap?
            # then just kinda... prepend it onto the next letter. this looks bad.
            next_char = all_chars[i+1]
            char_width = next_char[2][0] - next_char[1][0]
            new_ul = (max(next_char[1][0] - char_width, 0), next_char[1][1])
            new_lr = (max(next_char[2][0] - char_width, 0), next_char[2][1])
            align_transcript_chars.append([tra_char, new_ul, new_lr])

    #############################
    # -- GROUP INTO SYLLABLES --
    #############################

    syls = latsyl.syllabify_text(transcript)
    syls_boxes = []

    # get bounding boxes for each syllable
    syl_pos = -1                        # track of which syllable trying to get box of
    char_accumulator = ''               # check cur syl against this
    get_new_syl = True                  # flag that next loop should start a new syllable
    cur_ul = 0                          # upper-left point of last unassigned character
    cur_lr = 0                          # lower-right point of last character in loop
    for c in align_transcript_chars:    # @c can have more than one char in c[0].

        char_text = c[0].replace(' ', '')
        if not char_text:
            continue

        if get_new_syl:
            get_new_syl = False
            syl_pos += 1
            cur_syl = syls[syl_pos]
            cur_ul = c[1]

        # we'd rather not a syllable cross between lines. so, if it looks like that's about to happen
        # just forget about the part on the upper line and restart on the lower one.
        cur_lr = c[2]
        new_y_coord = c[1][1]
        if new_y_coord > cur_ul[1]:
            cur_ul = c[1]

        char_accumulator += char_text

        if verbose:
            print (cur_syl, char_accumulator, cur_ul, cur_lr)

        # if the accumulator has got the current syllable in it, remove the current syllable
        # from the accumulator and assign that syllable to the bounding box between cur_ul and cur_lr.
        # note that a syllable can be 'split,' in which case char_accumulator will have chars left in it
        if cur_syl in char_accumulator:
            char_accumulator = char_accumulator[len(cur_syl):]
            syls_boxes.append((cur_syl, cur_ul, cur_lr))
            get_new_syl = True

    return syls_boxes, image, lines_peak_locs


if __name__ == '__main__':

    import PIL
    from PIL import Image, ImageDraw, ImageFont

    fname = 'salzinnes_16'
    raw_image = gc.load_image('./png/' + fname + '_text.png')
    transcript = tsc.read_file('./png/' + fname + '_transcript.txt')
    syls_boxes, image, lines_peak_locs = process(raw_image, transcript, wkdir_name='test')

    #############################
    # -- DRAW RESULTS ON PAGE --
    #############################

    im = image.to_greyscale().to_pil()
    text_size = 70
    fnt = ImageFont.truetype('Arial.ttf', text_size)
    draw = ImageDraw.Draw(im)

    for i, char in enumerate(syls_boxes):
        if char[0] in '. ':
            continue

        ul = char[1]
        lr = char[2]
        draw.text((ul[0], ul[1] - text_size), char[0], font=fnt, fill='gray')
        draw.rectangle([ul, lr], outline='black')
        draw.line([ul[0], ul[1], ul[0], lr[1]], fill='black', width=10)

    for i, peak_loc in enumerate(lines_peak_locs):
        draw.text((1, peak_loc - text_size), 'line {}'.format(i), font=fnt, fill='gray')
        draw.line([0, peak_loc, im.width, peak_loc], fill='gray', width=3)

    im.save('testimg_{}.png'.format(fname))
    im.show()
