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
import json

reload(preproc)
reload(tsc)
reload(latsyl)

ocropus_model = './salzinnes_model-00054500.pyrnn.gz'
parallel = 2
median_line_mult = 2

# there are some hacks to make this work on windows (locally, not as a rodan job!).
# no guarantees, and OCRopus will throw out a lot of warning messages, but it works for local dev.
# you will have to use a model that is NOT zipped, or else go into the
# common.py file in ocrolib and change the way it's compressed from gunzip to gzip (gunzip is not
# natively available on windows). also, parallel processing will not work.
on_windows = (os.name == 'nt')


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
            return '{}: {}, {}'.format(self.char, self.ul, self.lr)
        else:
            return '{}: empty'.format(self.char)


def clean_special_chars(inp):
    '''
    removes some special characters from OCR output. ideally these would be useful but not clear how
    best to integrate them into the alignment algorithm. unidecode doesn't seem like these either
    '''
    inp = inp.replace('~', '')
    inp = inp.replace('\xc4\x81', 'a')
    inp = inp.replace('\xc4\x93', 'e')
    # there is no i with bar above in unicode (???)
    inp = inp.replace('\xc5\x8d', 'o')
    inp = inp.replace('\xc5\xab', 'u')
    return inp


def read_file(fname):
    '''
    helper function for reading a plaintext transcript of a manuscript page
    '''
    file = open(fname, 'r')
    lines = file.readlines()
    file.close()
    lines = ' '.join(x for x in lines if not x[0] == '#')
    lines = lines.replace('\n', '')
    lines = lines.replace('\r', '')
    lines = lines.replace('| ', '')
    # lines = unidecode(lines)
    return lines


def rotate_bbox(cbox, angle, orig_dim, target_dim, radians=False):
    pivot = gc.Point(orig_dim.ncols / 2, orig_dim.nrows / 2)

    # amount to translate to compensate for padding added by gamera's rotation in preprocessing
    dx = (orig_dim.ncols - target_dim.ncols) / 2
    dy = (orig_dim.nrows - target_dim.nrows) / 2

    if not radians:
        angle = angle * np.pi / 180

    s = np.sin(angle)
    c = np.cos(angle)

    # move to origin
    old_ulx = cbox.ulx - pivot.x
    old_uly = cbox.uly - pivot.y
    old_lrx = cbox.lrx - pivot.x
    old_lry = cbox.lry - pivot.y

    # rotate using a 2d rotation matrix
    new_ulx = (old_ulx * c) - (old_uly * s)
    new_uly = (old_ulx * s) + (old_uly * c)
    new_lrx = (old_lrx * c) - (old_lry * s)
    new_lry = (old_lrx * s) + (old_lry * c)

    # move back to original position adjusted for padding
    new_ulx += (pivot.x - dx)
    new_uly += (pivot.y - dy)
    new_lrx += (pivot.x - dx)
    new_lry += (pivot.y - dy)

    new_ul = np.round([new_ulx, new_uly]).astype('int16')
    new_lr = np.round([new_lrx, new_lry]).astype('int16')

    return CharBox(cbox.char, new_ul, new_lr)


def process(raw_image, transcript, wkdir_name='', parallel=parallel, median_line_mult=median_line_mult, ocropus_model=ocropus_model, verbose=True):
    '''
    given a text layer image @raw_image and a string transcript @transcript, performs preprocessing
    and OCR on the text layer and then aligns the results to the transcript text.
    '''

    #######################
    # -- PRE-PROCESSING --
    #######################

    # get raw image of text layer and preform preprocessing to find text lines
    # raw_image = gc.load_image('./png/' + filename + '_text.png')
    image, angle = preproc.preprocess_images(raw_image, None)
    cc_lines, lines_peak_locs, _ = preproc.identify_text_lines(image)

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
            cur_xpos = int(np.round(float(lsp[1]) + x_min))

            ul = (prev_xpos, y_min)
            lr = (cur_xpos, y_max)

            if lsp[0] == '~' or lsp[0] == '':
                new_box = CharBox(lsp[0], ul, lr)
                other_chars.append(new_box)
            else:
                # all_chars.append((clean_special_chars(lsp[0]), ul, lr))
                new_box = CharBox(clean_special_chars(lsp[0]), ul, lr)
                all_chars.append(new_box)

            prev_xpos = cur_xpos

    # delete working directory
    if on_windows:
        shutil.rmtree(dir)
    else:
        subprocess.check_call('rm -r ' + dir, shell=True)

    # get full ocr transcript
    ocr = ''.join(x.char for x in all_chars)
    all_chars_copy = list(all_chars)

    ###################################
    # -- PERFORM AND PARSE ALIGNMENT --
    ###################################
    tra_align, ocr_align = tsc.perform_alignment(transcript, ocr, verbose=verbose)

    align_transcript_boxes = []

    # insert gaps into ocr output based on alignment string. this causes all_chars to have gaps at the
    # same points as the ocr_align string does, and is thus the same length as tra_align.
    for i, char in enumerate(ocr_align):
        if char == '_':
            all_chars.insert(i, CharBox('_'))

    # this could very possibly go wrong (special chars, bug in alignment algorithm, etc) so better
    # make sure that this condition is holding at this point
    assert len(all_chars) == len(tra_align), 'all_chars not same length as alignment: ' \
        '{} vs {}'.format(len(all_chars), len(tra_align))

    for i, ocr_box in enumerate(all_chars):
        tra_char = tra_align[i]
        # print(len(align_transcript_boxes), tra_char, ocr_box)

        if not (tra_char == '_' or ocr_box.char == '_'):
            align_transcript_boxes.append(CharBox(tra_char, ocr_box.ul, ocr_box.lr))
        elif (tra_char != '_') and align_transcript_boxes:
            align_transcript_boxes[-1].char += tra_char
        elif (tra_char != '_'):
            # a tricky case: what if the first letter of the transcript is assigned to a gap?
            # then just kinda... prepend it onto the next letter. this looks bad.
            next_box = [x for x in all_chars[i:] if not x.char == '_'][0]
            char_width = next_box.width
            new_ul = (max(next_box.ulx - char_width, 0), next_box.uly)
            new_lr = (max(next_box.lrx - char_width, 0), next_box.lry)
            align_transcript_boxes.append(CharBox(tra_char, new_ul, new_lr))

    #############################
    # -- GROUP INTO SYLLABLES --
    #############################

    syls = latsyl.syllabify_text(transcript)
    syl_boxes = []

    # get bounding boxes for each syllable
    syl_pos = -1                        # track of which syllable trying to get box of
    char_accumulator = ''               # check cur syl against this
    get_new_syl = True                  # flag that next loop should start a new syllable
    cur_ul = 0                          # upper-left point of last unassigned character
    cur_lr = 0                          # lower-right point of last character in loop
    for box in align_transcript_boxes:    # @c can have more than one char in c[0].

        char_text = box.char.replace(' ', '')
        if not char_text:
            continue

        if get_new_syl:
            get_new_syl = False
            syl_pos += 1
            cur_syl = syls[syl_pos]
            cur_ul = box.ul

        # we'd rather not a syllable cross between lines. so, if it looks like that's about to happen
        # just forget about the part on the upper line and restart on the lower one.
        cur_lr = box.lr
        new_y_coord = box.uly
        if new_y_coord > cur_ul[1]:
            cur_ul = box.ul

        char_accumulator += char_text

        # if verbose:
        #     print(cur_syl, char_accumulator, cur_ul, cur_lr)

        # if the accumulator has got the current syllable in it, remove the current syllable
        # from the accumulator and assign that syllable to the bounding box between cur_ul and cur_lr.
        # note that a syllable can be 'split,' in which case char_accumulator will have chars left in it
        if cur_syl in char_accumulator:
            char_accumulator = char_accumulator[len(cur_syl):]
            syl_boxes.append(CharBox(cur_syl, cur_ul, cur_lr))
            get_new_syl = True

    # finally, rotate syl_boxes back by the angle that the page was rotated by
    for i in range(len(syl_boxes)):
        syl_boxes[i] = rotate_bbox(syl_boxes[i], -1 * angle, image.dim, raw_image.dim)

    return syl_boxes, image, lines_peak_locs


def to_JSON_dict(syl_boxes, lines_peak_locs):
    '''
    turns the output of the process script into a JSON dict that can be passed into the MEI_encoding
    rodan job.
    '''
    med_line_spacing = np.quantile(np.diff(lines_peak_locs), 0.75)

    data = {}
    data['median_line_spacing'] = med_line_spacing
    data['syl_boxes'] = []

    for s in syl_boxes:
        data['syl_boxes'].append({
            'syl': s.char,
            'ul': str(s.ul),
            'lr': str(s.lr)
        })

    return data


def draw_results_on_page(image, syl_boxes, lines_peak_locs):
    im = image.to_greyscale().to_pil()
    text_size = 50
    fnt = ImageFont.truetype('FreeMono.ttf', text_size)
    draw = ImageDraw.Draw(im)

    for i, cbox in enumerate(syl_boxes):
        if cbox.char in '. ':
            continue

        ul = cbox.ul
        lr = cbox.lr
        draw.text((ul[0], ul[1] - text_size), cbox.char, font=fnt, fill='gray')
        draw.rectangle([ul, lr], outline='black')
        draw.line([ul[0], ul[1], ul[0], lr[1]], fill='black', width=10)

    for i, peak_loc in enumerate(lines_peak_locs):
        draw.text((1, peak_loc - text_size), 'line {}'.format(i), font=fnt, fill='gray')
        draw.line([0, peak_loc, im.width, peak_loc], fill='gray', width=3)

    im.save('testimg_{}.png'.format(fname))
    # im.show()


if __name__ == '__main__':

    import PIL
    from PIL import Image, ImageDraw, ImageFont
    import os

    f_inds = [3]
    fnames = ['einsiedeln_00{}v'.format(f_ind) for f_ind in f_inds]

    for fname in fnames:
        text_layer_fname = './png/{}_text.png'.format(fname)
        transcript_fname = './png/{}_transcript.txt'.format(fname)

        if not os.path.isfile(text_layer_fname) or not os.path.isfile(transcript_fname):
            print('cannot find files for {}.'.format(fname))
            continue

        print('processing {}...'.format(fname))
        raw_image = gc.load_image('./png/' + fname + '_text.png')
        transcript = read_file('./png/' + fname + '_transcript.txt')
        syl_boxes, image, lines_peak_locs = process(raw_image, transcript, wkdir_name='test')

        # rot_img = raw_image.image_copy()
        # rot_img = rot_img.rotate(angle=angle, bgcolor=0)

        with open('{}.json'.format(fname), 'w') as outjson:
            json.dump(to_JSON_dict(syl_boxes, lines_peak_locs), outjson)
        draw_results_on_page(raw_image, syl_boxes, lines_peak_locs)
