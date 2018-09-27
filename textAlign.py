import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import itertools as iter
import functools
import alignmentGA
import os
import scipy.signal
import re
import bisect
import latinSyllabification
import textAlignPreprocessing as preproc
from os.path import isfile, join
import networkx as nx
import numpy as np
import syllable as syl
import PIL as pil
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
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


def plot(inp):
    plt.clf()
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


def draw_blob_alignment(alignment_groups, transcript_string,
                        cc_groups, gamera_image, size=45, fname='testimg.png'):
    '''
    visualizes the alignment given in @alignment_groups.
    '''
    gamera_image.save_image(fname)
    image = pil.Image.open(fname)
    draw = pil.ImageDraw.Draw(image)
    font = pil.ImageFont.truetype('Arial.ttf', size=size)

    cur_syl_index = 0
    cur_blob_index = 0
    for i in range(len(alignment_groups)):
        end_syl_index = cur_syl_index + alignment_groups[i][0]
        used_syllables_indices = list(range(cur_syl_index, end_syl_index))
        cur_syl_index = end_syl_index
        used_syllables = '-'.join([transcript_string[j] for j in used_syllables_indices])
        # used_syllables = str(alignment_groups[i][1]) + " " + used_syllables

        end_blob_index = cur_blob_index + alignment_groups[i][1]
        used_blob_indices = list(range(cur_blob_index, end_blob_index))
        cur_blob_index = end_blob_index
        used_blobs = [cc_groups[j] for j in used_blob_indices]

        for j, blob in enumerate(used_blobs):
            ul, lr = bounding_box(blob)
            position = (ul.x, ul.y - size)

            if j == 0:
                text = used_syllables if used_syllables else '(null)'
                text = str(alignment_groups[i][1]) + " " + text
            elif j == len(used_blobs) - 1:
                text = '->'
            else:
                text = '-'
            draw.text(position, text, fill='rgb(0, 0, 0)', font=font)

            # draw rectangle around this blob
            draw.rectangle((ul.x, ul.y, lr.x, lr.y), outline='rgb(0, 0, 0)')

    image.save(fname)


def alignment_fitness(alignment, blob_lengths, syl_lengths, gap_sizes):
    '''
    given an alignment between text blobs on the manuscript and syllables of the transcript,
    computes the error of the alignment based on the estimated syllable lengths and the known
    lengths present in the original image.
    '''
    cur_syl_pos = 0
    cur_blob_pos = 0
    cost = 0
    for i, x in enumerate(alignment):
        num_syls = x[0]
        num_blobs = x[1]

        # weight cost of each alignment element by space between blobs used, if more than one blob
        covered_gaps = [gap_sizes[i] for i in range(cur_blob_pos, cur_blob_pos + num_blobs - 1)]

        new_sum = sum(syl_lengths[cur_syl_pos:cur_syl_pos + num_syls])
        this_cost = (new_sum - sum(blob_lengths[cur_blob_pos:cur_blob_pos + num_blobs])) ** 2
        this_cost += sum(covered_gaps) ** 2

        cost += this_cost

        # update current position in sequence
        cur_syl_pos += num_syls
        cur_blob_pos += num_blobs

    return round(cost / cur_blob_pos)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_projection(strip):
    proj = strip.projection_cols()
    med = np.median([x for x in proj if x > 1])
    clipped = np.clip(proj, 0, med)
    max_clip = max(clipped)
    clipped = [x / max_clip for x in clipped]
    return clipped


def pos_align_fitness(positions, syllables, line_projs, verbose=False):
    word_min_gap = 30
    line_projs_flat = [item for sublist in line_projs for item in sublist]
    overlap_counter = [0] * (len(line_projs_flat) * 2)

    proj_lens = [len(x) for x in line_projs]
    line_break_positions = [sum(proj_lens[:x]) for x in range(len(proj_lens))]
    line_crosses_cost = 0
    word_min_gap_cost = 0

    score = 0
    for i, cur_pos in enumerate(positions):
        cur_end = cur_pos + syllables[i].width
        overlap_counter[cur_pos:cur_end] = [x + 1 for x in overlap_counter[cur_pos:cur_end]]

        # check if this syllable spans a line break - absolutely forbidden!
        crosses_line = [cur_pos < x < cur_end for x in line_break_positions]
        if any(crosses_line):
            line_crosses_cost += syllables[i].width

        if not syllables[i].word_begin:
            continue

        # if this syllable begins a word, then figure out what the gap is between the previous one
        # and heavily disincentivize small gaps
        left_bound = syllables[i - 1].width + positions[i - 1]
        gap = cur_pos - left_bound

        if gap < word_min_gap:
            word_min_gap_cost += word_min_gap - gap

    for i, val in enumerate(line_projs_flat):
        overlap_here = overlap_counter[i]
        if overlap_here == 0:
            score -= float(val)
        elif overlap_here == 1:
            score += float(val)
        else:
            score -= float(val) * overlap_here

    score -= word_min_gap_cost ** 2

    overlap_amt = sum([x * x for x in overlap_counter if x > 1.0])

    return (score,)


def absolute_to_relative_pos(position, strip_lengths):
    for i in range(len(strip_lengths)):
        if position > strip_lengths[i]:
            position -= strip_lengths[i]
        else:
            return i, position
    print('Given position larger than sum of strip lengths (out of bounds.)')
    return i, strip_lengths[-1]


def visualize_pos_align(positions, syllables, gamera_image, cc_strips, fname, size=30):

    strip_lengths = [x.ncols for x in cc_strips]

    gamera_image.save_image(fname)
    image = pil.Image.open(fname)
    draw = pil.ImageDraw.Draw(image)
    font = pil.ImageFont.truetype('Arial.ttf', size=size)

    # from current position of projection, find position on page
    for i, pos in enumerate(positions):
        end_pos = pos + syllables[i].width
        start_line, start_rel_pos = absolute_to_relative_pos(pos, strip_lengths)
        end_line, end_rel_pos = absolute_to_relative_pos(end_pos, strip_lengths)

        start_rel_pos += cc_strips[start_line].offset_x
        end_rel_pos += cc_strips[end_line].offset_x

        start_pt = (start_rel_pos, cc_strips[start_line].offset_y)
        end_pt = (end_rel_pos, cc_strips[end_line].offset_y)

        draw.line([start_pt, end_pt], fill='rgb(0, 0, 0)', width=5)
        draw.text(start_pt, syllables[i].text, fill='rgb(0, 0, 0)', font=font)

    image.save(fname)
    return


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

    for i in range(len(cc_groups)):

        cur_ccs = cc_groups[i]
        cur_syls = syl_groups[i]
        ul, lr = bounding_box(cur_ccs)

        draw.rectangle([ul.x, ul.y, lr.x, lr.y], fill=None, outline='rgb(0, 0, 0)')

        syl_text = '-'.join([x.text for x in cur_syls])
        syl_text = str(len(cur_ccs)) + ',' + str(len(cur_syls)) + ' ' + syl_text
        draw.text((ul.x, ul.y - size), syl_text, fill='rgb(0, 0, 0)', font=font)

    image.save(fname)


char_estimate_scale = 1.1

if __name__ == '__main__':
    # filename = 'salzinnes_11'
    # filename = 'einsiedeln_003v'
    # filename = 'stgall390_24'
    filename = 'klosterneuburg_23v'

    print('processing ' + filename + '...')

    raw_image = gc.load_image('./png/' + filename + '_text.png')
    try:
        staff_image = gc.load_image('./png/' + filename + '_stafflines.png')
    except IOError:
        staff_image = None
        print('no stafflines image...')

    image, staff_image = preproc.preprocess_images(raw_image, staff_image)
    cc_lines, lines_peak_locs = preproc.identify_text_lines(image)
    # cc_lines = preproc.find_ccs_under_staves(cc_lines, staff_image)
    cc_strips = [union_images(line) for line in cc_lines]
    line_projs = [normalize_projection(x) for x in cc_strips]

    # transcript_string contains each syllable. words_begin is 1 for every syllable that begins a
    # word and 0 for every syllable that does not
    transcript_string, words_begin = latinSyllabification.parse_transcript(
            './png/' + filename + '_transcript.txt')

    total_num_letters = sum([len(x) for x in transcript_string])

    # estimate width and volume of a single letter on average
    cc_lines_flat = [item for sublist in cc_lines for item in sublist]
    total_black = 0
    total_width = 0
    for cc in cc_lines_flat:
        total_width += cc.ncols
        total_black += cc.black_area()[0]

    avg_char_length = int((total_width / total_num_letters) * char_estimate_scale)
    avg_char_area = int(total_black / total_num_letters)

    words_end = [0] + words_begin[:-1]
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

    strip_total_length = sum([x.ncols for x in cc_strips])

    spaces = cc_scan_spacing(cc_lines_flat, image)
    new_image = draw_lines(image, lines_peak_locs)
    visualize_spacing(spaces, cc_lines_flat, new_image, 'testspacing.png')

    # BEGIN BRANCHING PROCEDURE

    sequences = [syl.AlignSequence()]
    # each sequence is a list of dicts each with three entries:
    # 'syl_groups' references the list of syllables for this element (0 to max_syls_per_element)
    # 'cc_groups' is the list of ccs that comprise this syllable (0 to forward_branches)
    # 'cost' is the cost associated with taking these ccs to estimate these syllables

    max_syls_per_element = 2
    max_ccs_per_element = 7
    max_sequences = 500
    diag_tol = 25

    # for step in range(2000):
    #     print('len seqs ' + str(len(sequences)))
    #     # get lowest (head, ccnums) for all sequences
    #     positions = [(x.last_cc_index(), x.last_syl_index()) for x in sequences]
    #     lowest_seq_pos = min(positions)
    #
    #     # find the best sequence in the lowest position and retain it for branching. remove all
    #     # other sequences with lowest position and discard them
    #     equiv_sequences = [x for x in sequences if (x.last_cc_index(), x.last_syl_index()) == lowest_seq_pos]
    #     best_seq = min(equiv_sequences, key=lambda x: sum(x.costs))
    #
    #     print('removing {}'.format(len(equiv_sequences)))
    #     for seq in equiv_sequences:
    #         sequences.remove(seq)
    #
    #     next_cc_ind = best_seq.last_cc_index() + 1
    #     next_syl_ind = best_seq.last_syl_index() + 1
    #
    #     this_max_ccs = min(max_ccs_per_element, len(cc_lines_flat) - next_cc_ind)
    #     this_max_syls = min(max_syls_per_element, len(syllables) - next_syl_ind)
    #
    #     # get pairs of (num ccs, num syls) to define new elements to append to this one
    #     arrangements = iter.product(range(1, this_max_ccs + 1), range(this_max_syls + 1))
    #
    #     for i in arrangements:
    #         num_syls = i[1]
    #         num_ccs = i[0]
    #         add_cc_group = cc_lines_flat[next_cc_ind:next_cc_ind + num_ccs]
    #         add_spaces = spaces[next_cc_ind:next_cc_ind + num_ccs]
    #         add_syl_group = syllables[next_syl_ind:next_syl_ind + num_syls]
    #
    #         # a syllable that starts a word must be at the beginning of a syl group, and one that
    #         # ends a word must be at the END of a syl group. if this is not the case, then skip
    #         # this possible arrangement
    #
    #         syl_begins = [x.word_begin for x in add_syl_group]
    #         syl_ends = [x.word_end for x in add_syl_group]
    #         if any(syl_begins[1:]) or any(syl_ends[:-1]):
    #             # print('begin / end violation')
    #             continue
    #
    #         # make a new sequence out of this branched one
    #         cost = syl.get_cost_of_element(add_cc_group, add_syl_group, add_spaces)
    #         new_cc_groups = best_seq.cc_groups + [add_cc_group]
    #         new_syl_groups = best_seq.syl_groups + [add_syl_group]
    #         new_costs = best_seq.costs + [cost]
    #
    #         new_seq = syl.AlignSequence(syl_groups=new_syl_groups,
    #                                     cc_groups=new_cc_groups,
    #                                     costs=new_costs)
    #
    #         sequences.append(new_seq)
    #
    #         if len(sequences) <= max_sequences:
    #             continue
    #
    #         sequences.sort(key=lambda x: np.average(x.costs))
    #         sequences = sequences[:max_sequences]
    # visualize_alignment(sequences[0], new_image, 'testalign.png')

    num_ccs = len(cc_lines_flat)
    num_syls = len(syllables)
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

            cost = syl.get_cost_of_element(ccs, syls, sps)
            edges.append((node, n, cost))

        g.add_weighted_edges_from(edges)

    path = nx.dijkstra_path(g, (0, 0), max(nodes))
    best_seq = syl.make_align_seq_from_path(path, cc_lines_flat, syllables)
    visualize_alignment(best_seq, image, 'testalign.png')
