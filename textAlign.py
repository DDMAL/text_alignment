import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import itertools as iter
import functools
import os
import re
import latinSyllabification
import textAlignPreprocessing as preproc
from os.path import isfile, join
import networkx as nx
import numpy as np
import syllable as syl
import PIL as pil
from PIL import Image, ImageDraw, ImageFont
reload(latinSyllabification)
reload(preproc)
reload(syl)
# reload(alignmentGA)


def bounding_box(cc_list):
    '''
    given a list of connected components, finds the smallest
    bounding box that encloses all of them.
    '''

    upper = [x.offset_y for x in cc_list]
    lower = [x.offset_y + x.nrows for x in cc_list]
    left = [x.offset_x for x in cc_list]
    right = [x.offset_x + x.ncols for x in cc_list]

    ul = gc.Point(min(left), min(upper))
    lr = gc.Point(max(right), max(lower))

    return ul, lr


def imsv(img, fname="testimg.png"):
    if type(img) == list:
        union_images(img).save_image(fname)
    else:
        img.save_image(fname)


def plot(inp, srt=False):
    plt.clf()
    if srt:
        inp = sorted(inp)
    plt.figure(num=None, dpi=400, figsize=(30, 3))
    plt.plot(inp, c='black', linewidth=0.5)
    plt.savefig("testplot.png")


def draw_lines(image, line_locs, horizontal=True):
    new = image.image_copy()
    for l in line_locs:
        if horizontal:
            start = gc.FloatPoint(0, l)
            end = gc.FloatPoint(image.ncols, l)
        else:
            start = gc.FloatPoint(l, 0)
            end = gc.FloatPoint(l, image.nrows)
        new.draw_line(start, end, 1, 2)
    return new


def cc_scan_spacing(cc_lines_flat, image, max_lookahead=5):
    spaces = []

    # each entry in spaces[] will be the size of the space between the cc at position i and the
    # cc at position i+1
    for ind in range(len(cc_lines_flat) - 1):

        left_cc = cc_lines_flat[ind]
        left_contour = list(left_cc.contour_right())
        left_range = [left_cc.offset_y, left_cc.offset_y + left_cc.nrows]

        # keep looking forward until we reach a point where the next cc is behind left_cc OR we just
        # reach the end of cc_lines_flat
        lookahead = 1
        while (lookahead < max_lookahead and
                ind + lookahead < len(cc_lines_flat) and
                cc_lines_flat[ind + lookahead].lr.x > left_cc.ul.x):
            lookahead += 1

        if lookahead >= 2:
            ul, lr = bounding_box(cc_lines_flat[ind + 1:ind + lookahead])
            right_cc = image.subimage(ul, lr)
        else:
            right_cc = cc_lines_flat[ind + 1]

        right_range = [right_cc.offset_y, right_cc.offset_y + right_cc.nrows]
        right_contour = list(right_cc.contour_left())

        # pad the top of each contour
        top_diff = left_range[0] - right_range[0]
        if top_diff < 0:
            right_contour = ([np.inf] * abs(top_diff)) + right_contour
        elif top_diff > 0:
            left_contour = ([np.inf] * abs(top_diff)) + left_contour

        # pad the bottom of each contour
        bottom_diff = left_range[1] - right_range[1]
        if bottom_diff < 0:
            left_contour = left_contour + ([np.inf] * abs(bottom_diff))
        elif bottom_diff > 0:
            right_contour = right_contour + ([np.inf] * abs(bottom_diff))

        # horizontal space between the two ccs' bounding boxes
        cc_bb_space = right_cc.offset_x - (left_cc.offset_x + left_cc.ncols)

        # add contours of left and right to get space between them
        cc_row_spacing = [left_contour[i] + right_contour[i] for i in range(len(left_contour))]
        cc_row_spacing = [x + cc_bb_space for x in cc_row_spacing if not x == np.inf]
        if cc_row_spacing:
            med = int(np.median(cc_row_spacing))
            med = max(med, 0)
            spaces.append(med)
            continue

        # if not cc_row_spacing then the two ccs do not overlap horizontally
        # check to see if they overlap vertically

        left_midpoint = left_cc.offset_x + int(left_cc.ncols / 2)
        right_midpoint = right_cc.offset_x + int(right_cc.ncols / 2)
        diff = right_midpoint - left_midpoint
        if diff >= 0:
            spaces.append(diff)
        else:
            spaces.append(np.inf)

    return spaces


def visualize_spacing(gap_sizes, cc_lines_flat, gamera_image, fname, size=14):

    gamera_image.save_image(fname)
    image = pil.Image.open(fname)
    draw = pil.ImageDraw.Draw(image)
    font = pil.ImageFont.truetype('Arial.ttf', size=size)

    for i, cc in enumerate(cc_lines_flat):

        draw.rectangle([cc.ul.x, cc.ul.y, cc.lr.x, cc.lr.y], fill=None, outline='rgb(0, 0, 0)')

        if i == len(cc_lines_flat) - 1:
            continue

        # draw estimated size of gap directly over gap
        draw.text((int((cc.ul.x + cc.lr.x) / 2), cc.ul.y - size), str(gap_sizes[i]), fill='rgb(0, 0, 0)', font=font)

    image.save(fname)


def visualize_alignment(sequence, gamera_image, fname, size=20):

    gamera_image.save_image(fname)
    image = pil.Image.open(fname)
    draw = pil.ImageDraw.Draw(image)
    font = pil.ImageFont.truetype('Arial.ttf', size=size)

    cc_groups = sequence.cc_groups
    syl_groups = sequence.syl_groups
    costs = sequence.costs

    for i in range(len(cc_groups)):

        cur_ccs = cc_groups[i]
        cur_syls = syl_groups[i]
        ul, lr = bounding_box(cur_ccs)

        draw.rectangle([ul.x, ul.y, lr.x, lr.y], fill=None, outline='rgb(0, 0, 0)')

        syl_text = '-'.join([x.text for x in cur_syls])
        syl_text = str(len(cur_ccs)) + ',' + str(len(cur_syls)) + ' ' + syl_text
        syl_text = str(costs[i]) + '\n' + syl_text
        draw.text((ul.x, ul.y - size*2), syl_text, fill='rgb(0, 0, 0)', font=font)

    image.save(fname)


if __name__ == '__main__':
    # filename = 'salzinnes_11'
    # filename = 'einsiedeln_003v'
    filename = 'stgall390_23'
    # filename = 'klosterneuburg_23v'

    print('processing ' + filename + '...')

    raw_image = gc.load_image('./png/' + filename + '_text.png')
    try:
        staff_image = gc.load_image('./png/' + filename + '_stafflines.png')
    except IOError:
        staff_image = None
        print('no stafflines image...')

    image, staff_image = preproc.preprocess_images(raw_image, staff_image)
    cc_lines, lines_peak_locs = preproc.identify_text_lines(image)
    cc_lines = preproc.find_ccs_under_staves(cc_lines, staff_image)
    cc_strips = [union_images(line) for line in cc_lines]

    cc_lines_flat = [item for sublist in cc_lines for item in sublist]

    # transcript_string contains each syllable. words_begin is 1 for every syllable that begins a
    # word and 0 for every syllable that does not
    transcript_string, words_begin = latinSyllabification.parse_transcript(
            './png/' + filename + '_transcript.txt')

    # estimate width and volume of a single letter on average
    total_num_letters = sum([len(x) for x in transcript_string])
    median_area = np.median([x.black_area()[0] for x in cc_lines_flat])
    total_black = 0
    total_width = 0
    for cc in cc_lines_flat:
        total_width += cc.ncols
        total_black += min(cc.black_area()[0], median_area)

    avg_char_length = int((total_width / total_num_letters))
    avg_char_area = int(total_black / total_num_letters)

    # syllables contains a list of all syllables in the transcript with estimates of area and width
    words_end = words_begin[1:] + [1]
    syllables = []
    for i in range(len(transcript_string)):
        width = avg_char_length * len(transcript_string[i])
        area = avg_char_area * len(transcript_string[i])
        syllables.append(syl.Syllable(
            text=transcript_string[i],
            word_begin=words_begin[i],
            word_end=words_end[i],
            width=width,
            area=area)
            )

    num_ccs = len(cc_lines_flat)
    num_syls = len(syllables)

    # spaces finds the vertically-scanned space between all consecutive pairs of ccs in
    # cc_lines_flat
    spaces = cc_scan_spacing(cc_lines_flat, image)
    median_space = np.median(spaces)

    new_image = draw_lines(image, lines_peak_locs)
    visualize_spacing(spaces, cc_lines_flat, new_image, 'testspacing.png')

    # estimate max_ccs_per_element: calculate about how many syllables there are per cc, and round
    # that estimate up generously
    # max_ccs_per_element = 10
    max_ccs_per_element = np.ceil((float(num_ccs) / num_syls) * 3)
    max_syls_per_element = 2
    diag_tol = 15

    # the nodes of the sequence alignment graph should be the superdiagonals of a rectangular
    # matrix. use diag_tol as the width of the strip down the diagonal
    nodes = iter.product(range(num_ccs), range(num_syls))
    slope = float(num_syls) / num_ccs
    nodes = [x for x in nodes if abs(slope * x[0] - x[1]) < diag_tol]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    # for each node in nodes, connect it to the succeeding rectangle
    print('building graph...')

    for i, node in enumerate(nodes):

        if i % 1000 == 0:
            print('    branching node {} of {}...').format(i, len(nodes))

        # nodes in @node's successor rectangle. using <= for syllables, because it is possible to
        # assign 0 syllables to an element, whereas at least 1 cc must be in each element
        successor_nodes = [x for x in nodes if
            (0 < x[0] - node[0] <= max_ccs_per_element) and
            (0 <= x[1] - node[1] <= max_syls_per_element)]

        edges = []
        for n in successor_nodes:
            ccs = cc_lines_flat[node[0]:n[0]]
            sps = spaces[node[0]:n[0]]
            syls = syllables[node[1]:n[1]]

            # WORD_BEGIN syllables can only be at the BEGINNING of elements,
            # and WORD_END syllables can only be at the END of elements.
            if len(syls) > 1 and \
                    (any(x.word_begin for x in syls[1:]) or
                    any(x.word_end for x in syls[:-1])):
                continue

            cost = syl.get_cost_of_element(ccs, syls, sps, median_area, median_space)
            edges.append((node, n, cost))

        g.add_weighted_edges_from(edges)

    print('finding optimal path thru graph...')
    path = nx.dijkstra_path(g, (0, 0), max(nodes))
    best_seq = syl.make_align_seq_from_path(path, cc_lines_flat, syllables, spaces)
    visualize_alignment(best_seq, image, 'testalign.png')
