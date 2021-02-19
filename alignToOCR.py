# -*- coding: utf-8 -*-

# import gamera.core as gc
# gc.init_gamera()
# from gamera.plugins.image_utilities import union_images
import matplotlib.pyplot as plt
import textAlignPreprocessing as preproc
import os
import shutil
import numpy as np
import affine_needleman_wunsch as afw
import latinSyllabification as latsyl
import parse_cantus_csv as pcc
import subprocess
import json
import re
import io
import tempfile
import cv2 as cv


from importlib import reload
reload(preproc)
reload(afw)
reload(latsyl)
reload(pcc)

parallel = 2
median_line_mult = 2

# there are some hacks to make this work on windows (locally, not as a rodan job!).
# no guarantees, and OCRopus will throw out a lot of warning messages, but it works for local dev.
# you will have to use a model that is NOT zipped, or else go into the
# common.py file in ocrolib and change the way it's compressed from gunzip to gzip (gunzip is not
# natively available on windows). also, parallel processing will not work.
on_windows = (os.name == 'nt')


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

    # amount to translate to compensate for padding added by gamera's rotation in preprocessing.
    # i am pretty sure this is the most "correct" amount. my math might be off.
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


def process(raw_image,
            transcript,
            ocropus_model,
            seq_align_params={},
            wkdir_name='wkdir_ocropy',
            parallel=parallel,
            median_line_mult=median_line_mult,
            existing_ocr=None,
            verbose=True):
    '''
    given a text layer image @raw_image and a string transcript @transcript, performs preprocessing
    and OCR on the text layer and then aligns the results to the transcript text.
    '''

    #######################
    # -- PRE-PROCESSING --
    #######################

    # get raw image of text layer and preform preprocessing to find text lines
    print('identifying text lines...')
    image, eroded, angle = preproc.preprocess_images(raw_image)
    cc_strips, lines_peak_locs, _ = preproc.identify_text_lines(eroded)

    #################################
    # -- PERFORM OCR WITH OCROPUS --
    #################################

    all_chars = existing_ocr

    if not all_chars:
        all_chars = perform_ocr.recognize_text_strips(image, cc_strips, ocropus_model)

    #############################
    # -- HANDLE ABBREVIATIONS --
    #############################
    print('handling abbreviations...')

    abbreviations = latsyl.abbreviations
    for abb in abbreviations.keys():
        while True:
            ocr_str = ''.join(unicode(x.char) for x in all_chars)
            idx = ocr_str.find(abb)

            if idx == -1:
                break
            ins = []

            for i, segment in enumerate(abbreviations[abb]):
                split_box = all_chars[i + idx]
                ins += [CharBox(x, split_box.ul, split_box.lr) for x in segment]
            all_chars = all_chars[:idx] + ins + all_chars[idx + len(abb):]

    # get full ocr transcript
    ocr = ''.join(x.char for x in all_chars)
    all_chars_copy = list(all_chars)

    ###################################
    # -- PERFORM AND PARSE ALIGNMENT --
    ###################################

    # remove special characters, but maintain case
    transcript = pcc.clean(transcript)

    print('performing alignment...')
    tra_align, ocr_align, _ = afw.perform_alignment(
        transcript=list(transcript),
        ocr=list(ocr),
        **seq_align_params
        )
    tra_align = ''.join(tra_align)
    ocr_align = ''.join(ocr_align)

    print('syllabifying...')
    syls = latsyl.syllabify_text(transcript)
    align_transcript_boxes = []
    current_offset = 0
    syl_boxes = []

    print('matching syllables to alignment...')
    # insert gaps into ocr output based on alignment string. this causes all_chars to have gaps at the
    # same points as the ocr_align string does, and is thus the same length as tra_align.
    for i, char in enumerate(ocr_align):
        if char == '_':
            all_chars.insert(i, CharBox('_'))

    # this could very possibly go wrong (special chars, bug in alignment algorithm, etc) so better
    # make sure that this condition is holding at this point
    assert len(all_chars) == len(tra_align), 'all_chars not same length as alignment: ' \
        '{} vs {}'.format(len(all_chars), len(tra_align))

    # for each syllable in the transcript, find what characters (or gaps) of the ocr that syllable
    # is aligned to.
    for syl in syls:
        if len(syl) < 1:
            continue
        elif len(syl) == 1:
            syl_regex = syl
        else:
            syl_regex = syl[0] + syl[1:-1].replace('', '_*') + syl[-1]

        syl_match = re.search(syl_regex, tra_align[current_offset:], re.IGNORECASE)
        start = syl_match.start() + current_offset
        end = syl_match.end() + current_offset
        current_offset = end
        align_boxes = [x for x in all_chars[start:end] if x.lr is not None]

        # if align_boxes is empty then this syllable got aligned to nothing in the ocr. ignore it.
        if not align_boxes:
            continue

        # if align_boxes has boxes that lie on multiple text lines then we're trying to align this
        # single syllable over multiple lines. remove all boxes on the upper line.
        if len(set([x.uly for x in align_boxes])) > 1:
            lower_level = max(x.uly for x in align_boxes)
            align_boxes = [b for b in align_boxes if b.uly == lower_level]

        new_ul = (min(x.ulx for x in align_boxes), min(x.uly for x in align_boxes))
        new_lr = (max(x.lrx for x in align_boxes), max(x.lry for x in align_boxes))
        syl_boxes.append(CharBox(syl, new_ul, new_lr))

    print('rotating bboxes again...')
    # finally, rotate syl_boxes back by the angle that the page was rotated by
    for i in range(len(syl_boxes)):
        syl_boxes[i] = rotate_bbox(syl_boxes[i], -1 * angle, image.dim, raw_image.dim)

    return syl_boxes, image, lines_peak_locs, all_chars_copy


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
            'ul': [int(s.ul[0]), int(s.ul[1])],
            'lr': [int(s.lr[0]), int(s.lr[1])]
        })

    return data


def draw_results_on_page(image, syl_boxes, lines_peak_locs):
    im = image.to_greyscale().to_pil()
    text_size = image.ncols // 74
    fnt = ImageFont.truetype('FreeMono.ttf', text_size)
    draw = ImageDraw.Draw(im)

    for i, cbox in enumerate(syl_boxes):
        if cbox.char in '. ':
            continue

        ul = cbox.ul
        lr = cbox.lr
        draw.rectangle([ul[0], ul[1] - text_size, ul[0] + text_size * len(cbox.char) * 0.6, ul[1]], fill='white')
        draw.text((ul[0], ul[1] - text_size), cbox.char, font=fnt, fill='black')
        draw.rectangle([ul, lr], outline='black')
        draw.line([ul[0], ul[1], ul[0], lr[1]], fill='black', width=5)

    # for i, peak_loc in enumerate(lines_peak_locs):
    #     draw.text((1, peak_loc - text_size), 'line {}'.format(i), font=fnt, fill='gray')
    #     draw.line([0, peak_loc, im.width, peak_loc], fill='gray', width=3)

    im.save('./out_imgs/{}_alignment.png'.format(fname))
    # im.show()


if __name__ == '__main__':

    import parse_cantus_csv as pcc
    reload(pcc)
    import PIL
    import pickle
    from PIL import Image, ImageDraw, ImageFont
    import os

    text_func = pcc.filename_to_text_func('./csv/123723_Salzinnes.csv', './csv/mapping.csv')
    manuscript = 'salzinnes'
    f_inds = ['040r']
    ocropus_model = './salzinnes_model-00054500.pyrnn.gz'

    # text_func = pcc.filename_to_text_func('./csv/einsiedeln_123606.csv')
    # manuscript = 'einsiedeln'
    # f_inds = range(0, 11)
    # ocropus_model = './salzinnes_model-00054500.pyrnn.gz'

    # text_func = pcc.filename_to_text_func('./csv/stgall390_123717.csv')
    # manuscript = 'stgall390'
    # f_inds = ['022', '023', '024', '025', '007']
    # ocropus_model = 'stgall2-00017000.pyrnn.gz'

    # text_func = pcc.filename_to_text_func('./csv/stgall388_123750.csv')
    # manuscript = 'stgall388'
    # f_inds = ['028', '029', '030', '031', '032']
    # ocropus_model = 'stgall3-00017000.pyrnn.gz'

    for ind in f_inds:

        try:
            f_id, transcript = text_func(ind)
        except ValueError as e:
            print(e)
            print('no chants listed for page {}'.format(ind))
            continue

        fname = f'{manuscript}_{f_id}'
        ocr_pickle = None  # './pik/{}_boxes.pickle'.format(fname)
        text_layer_fname = f'./png/{fname}_text.png'

        if not os.path.isfile(text_layer_fname):
            print('cannot find files for {}.'.format(fname))
            continue

        print('processing {}...'.format(fname))
        raw_image = cv.imread(f'./png/{fname}_text.png')

        existing_ocr = None
        # if ocr_pickle:
        #     try:
        #         with open(ocr_pickle) as f:
        #             existing_ocr = pickle.load(f)
        #         print('using pickled ocr results in {}...'.format(ocr_pickle))
        #     except IOError:
        #         print('Pickle file {} not found - performing ocr instead'.format(ocr_pickle))
        #     except AttributeError:
        #         print('Pickle error: re-performing ocr')

        id = hex(np.random.randint(2**32))
        result = process(raw_image, transcript, ocropus_model,
            wkdir_name='ocr_{}'.format(id), existing_ocr=existing_ocr, verbose=True)
        if result is None:
            continue
        syl_boxes, image, lines_peak_locs, all_chars = result
        with open('./out_json/{}.json'.format(fname), 'w') as outjson:
            json.dump(to_JSON_dict(syl_boxes, lines_peak_locs), outjson)
        with open('./pik/{}_boxes.pickle'.format(fname), 'wb') as f:
            pickle.dump(all_chars, f, -1)

        draw_results_on_page(raw_image, syl_boxes, lines_peak_locs)
