import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import itertools as iter
import functools
import alignmentGA
import os
import re
import bisect
import latinSyllabification
import textAlignPreprocessing as preproc
from os.path import isfile, join
import numpy as np
import syllable as syl
import PIL as pil  # python imaging library, for testing only
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
reload(latinSyllabification)
reload(preproc)
reload(syl)
reload(alignmentGA)


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

    line_projs_flat = [item for sublist in line_projs for item in sublist]
    overlap_counter = [0] * (len(line_projs_flat) * 2)

    proj_lens = [len(x) for x in line_projs]
    line_break_positions = [sum(proj_lens[:x]) for x in range(len(proj_lens))]
    line_crosses_cost = 0

    score = 0
    for i, cur_pos in enumerate(positions):
        cur_end = cur_pos + syllables[i].width
        overlap_counter[cur_pos:cur_end] = [x + 1 for x in overlap_counter[cur_pos:cur_end]]

        # check if this syllable spans a line break - absolutely forbidden!
        crosses_line = [cur_pos < x < cur_end for x in line_break_positions]
        if any(crosses_line):
            line_crosses_cost += syllables[i].width

    for i, val in enumerate(line_projs_flat):
        overlap_here = overlap_counter[i]
        if overlap_here == 0:
            score -= float(val)
        elif overlap_here == 1:
            score += float(val)
        else:
            score -= overlap_here * overlap_here

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
    position = 0
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
        position = end_pos

    image.save(fname)
    return

char_estimate_scale = 0.8

if __name__ == '__main__':
    filename = 'salzinnes_18'
    # filename = 'einsiedeln_002v'
    # filename = 'stgall390_07'
    # filename = 'klosterneuburg_23v'

    # def process(filename):
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

    syllables = []
    for i in range(len(transcript_string)):
        width = avg_char_length * len(transcript_string[i])
        syllables.append(syl.Syllable(
            text=transcript_string[i],
            word_begin=words_begin[i],
            width=width)
            )

    fitness_func = functools.partial(pos_align_fitness, syllables=syllables, line_projs=line_projs)
    strip_total_length = sum([x.ncols for x in cc_strips])
    # pop, log, hof = alignmentGA.run_GA(fitness_func, len(syllables), strip_total_length)
    # visualize_pos_align(hof[0], syllables, image, cc_strips, 'testimg align.png')
    # test_positions = [100, 101, 500, 700, 1200, 1400, 1700, 2300, 2500, 2800, 4900]
    # res = pos_align_fitness(test_positions, syllables, line_projs)
    # print(res)

    # TEST WITH FORWARD CONVOLUTION

lookahead_pixels = 1500
branches_per_step = 6
max_num_seqs = 125

completed_sequences = []

print('precomputing convolutions...')
seqs = [syl.AlignSequence(positions=[])]
convolutions = {}
for width in set([s.width for s in syllables]):
    conv = [sum(line_projs_flat[i:i+width]) for i in range(strip_total_length)]
    convolutions[width] = conv

for step in range(3000):

    if not seqs:
        print('finished branching.')
        break

    print(step, len(seqs), max([x.score for x in seqs]), len([x for x in seqs if x.completed]))
    this_equiv = seqs[0].equivalence()

    equivalent_seqs = [s for s in seqs if s.equivalence() == this_equiv]
    current_seq = max(equivalent_seqs, key=lambda x: x.score)

    # remove equivalent seqs; we only care about the one with highest score
    print('removing ' + str(len(equivalent_seqs)) + ' from class ' + str(this_equiv))
    for seq in equivalent_seqs:
        seqs.remove(seq)

    current_head = current_seq.head()
    width = syllables[len(current_seq.positions)].width
    convolve_slice = convolutions[width][current_head:current_head + lookahead_pixels]

    # get the most prominent peaks in the lookahead interval of the convolution
    peaks = preproc.find_peak_locations(convolve_slice, 0, ranked=True)

    if (current_head + width > strip_total_length or
            len(syllables) == len(current_seq.positions) or
            not peaks):
        current_seq.completed = True
        completed_sequences.append(current_seq)
        continue

    # adding current_head here to make sure we're correctly aligned with the global line projection
    next_locs = [x[0] + current_head for x in peaks[:branches_per_step]]

    # make new sequences using the next_locs, evaluate their scores, and insert them into the sequence
    # list, using bisect to keep it sorted so that the ones with the lowest head() are always first
    for loc in next_locs:
        new_seq = syl.AlignSequence(positions=current_seq.positions + [loc])
        new_seq.score = pos_align_fitness(new_seq.positions, syllables, line_projs)[0]
        keys = [s.head() for s in seqs]
        ind = bisect.bisect_left(keys, loc)
        seqs.insert(ind, new_seq)

    if len(seqs) > max_num_seqs:
        scores = sorted([x.score for x in seqs], reverse=True)
        seqs = [x for x in seqs if x.score > scores[max_num_seqs]]

print(seqs)

plt.clf()
plt.figure(num=None, dpi=400, figsize=(20, 3))
plt.plot(convolved, c='black', linewidth=0.5)
plt.plot(line_projs_flat[0:lookahead_pixels], c='gray', linewidth=0.5)
plt.savefig("testplot.png")
