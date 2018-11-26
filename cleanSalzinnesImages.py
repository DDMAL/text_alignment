from os.path import isfile, join
import numpy as np
import PIL as pil  # python imaging library, for testing only
import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import itertools as iter
import os
import re
import textAlignPreprocessing as preproc
reload(preproc)

despeckle_amt = 25
filename = 'salzinnes_11'
pad_amt = 100


def clean_image(input_image, despeckle_amt=despeckle_amt, filter_runs=1, filter_runs_amt=1, cc_min_size=50):
    '''
    modified version of preprocess_images from the preprocessing file; just intended to get
    salzinnes in better shape for OCRopus training
    '''
    image_bin = input_image.to_onebit()
    ccs = image_bin.cc_analysis()
    for c in ccs:
        area = c.black_area()[0]
        if area < cc_min_size:
            c.fill_white()

    image_bin.invert()
    image_bin.despeckle(despeckle_amt)
    image_bin.invert()
    image_bin.reset_onebit_image()

    # find likely rotation angle and correct
    angle, tmp = image_bin.rotation_angle_projections()
    image_bin = image_bin.rotate(angle=angle)

    for i in range(filter_runs):
        image_bin.filter_short_runs(filter_runs_amt, 'black')
        image_bin.filter_narrow_runs(filter_runs_amt, 'black')

    return image_bin


if __name__ == '__main__':

    all_files = os.listdir('./png/')
    fnames = [x for x in all_files if 'text.png' in x and 'salzinnes' in x]

    for filename in fnames:
        print('processing ' + filename + '...')

        raw_image = gc.load_image('./png/' + filename)
        img = clean_image(raw_image)
        cc_lines, peaks_locs = preproc.identify_text_lines(img)
        cc_lines_flat = [item for sublist in cc_lines for item in sublist]
        unioned_lines = union_images(cc_lines_flat)
        # unioned_lines = unioned_lines.pad_image(pad_amt, pad_amt, pad_amt, pad_amt)
        unioned_lines.save_image('cleaned_' + filename)
