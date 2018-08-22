import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import itertools as iter
import os
import re
import latinSyllabification
import textAlignPreprocessing as preproc
from os.path import isfile, join
import numpy as np
import PIL as pil  # python imaging library, for testing only
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
reload(latinSyllabification)
reload(preproc)

# PARAMETERS FOR PREPROCESSING
# saturation_thresh = 0.6
# sat_area_thresh = 150
# despeckle_amt = 100            # an int in [1,100]: ignore ccs with area smaller than this
# noise_area_thresh = 600        # an int in : ignore ccs with area smaller than this
#
# # PARAMETERS FOR TEXT LINE SEGMENTATION
# filter_size = 20                # size of moving-average filter used to smooth projection
# prominence_tolerance = 0.50     # log-projection peaks must be at least this prominent
# collision_strip_size = 50       # in [0,inf]; amt of each cc to consider when clipping
# remove_capitals_scale = 2
#
# # CC GROUPING (BLOBS)
# cc_group_gap_min = 10  # any gap at least this wide will be assumed to be a space between words!

max_blob_sequences = 1000  # so nothing gets stuck
num_blobs_lookahead = 3

letter_width_dict = {
    '*': 2,
#     'm': 128,
#     'l': 36,
#     'i': 36,
#     'a': 84,
#     'c': 60,
#     'e': 59,
 }


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
    asdf = plt.plot(inp, c='black', linewidth=0.5)
    plt.savefig("testplot.png", dpi=800)


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


if __name__ == "__main__":
    single = True
    # filename = 'salzinnes_24'
    # filename = 'einsiedeln_002v'
    filename = 'stgall390_07'

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

    gap_sizes = []
    cc_groups = []
    for x in cc_lines:
        grouped, gaps = preproc.group_ccs(x)
        gap_sizes += gaps + [np.inf]
        cc_groups += grouped

    transcript_string, words_begin = latinSyllabification.parse_transcript(
            './png/' + filename + '_transcript.txt')

    blob_lengths = []
    for g in cc_groups:
        width = sum([x.ncols for x in g])
        blob_lengths.append(width)

    transcript_lengths = [len(x) for x in transcript_string]

    # estimate length of each syllable
    avg_char_length = round(sum(blob_lengths) / sum(transcript_lengths))
    avg_syl_length = round(sum(blob_lengths) / len(blob_lengths))
    syl_lengths = []
    letter_dict = defaultdict(lambda: avg_char_length, **letter_width_dict)
    for syl in transcript_string:
        this_width = 0
        for char in syl:
            this_width += letter_dict[char]
        syl_lengths.append(this_width)

    # set up for sequence searching
    first_word_begin = [x for x in words_begin if x > 0][0]
    init_seqs = iter.product(range(0, first_word_begin + 1), range(1, num_blobs_lookahead + 1))
    sequences = [[x] for x in init_seqs]
    finished_seqs = []

    def current_position_of_seq(seq):
        return (sum([x[0] for x in seq]), sum([x[1] for x in seq]))

    # iterating over blobs
    continue_looping = True
    while(sequences):

        print(sequences[0])

        new_sequences = []
        # print('group length', gl)

        # branch out from every sequence in list of sequences
        blob_min_pos = min([sum([x[1] for x in seq]) for seq in sequences])
        print('min blob position', blob_min_pos)
        earliest_seqs = []
        other_seqs = []
        # earliest_seqs = [seq for seq in sequences if sum([x[1] for x in seq]) == blob_min_pos]
        for seq in sequences:
            if sum([x[1] for x in seq]) == blob_min_pos:
                earliest_seqs.append(seq)
            else:
                other_seqs.append(seq)

        print('earliest seqs len', len(earliest_seqs), 'vs', len(other_seqs))

        branches = []
        for seq in earliest_seqs:

            # lower bound on number of syllables that could possibly be assigned to this blob?
            min_branches = 0

            # max number of branches = branch syllables until reach end of current word
            pos, blob_pos = current_position_of_seq(seq)

            if pos == len(transcript_string) or blob_pos == len(blob_lengths):
                this_fitness = alignment_fitness(seq, blob_lengths, syl_lengths, gap_sizes)
                finished_seqs.append((this_fitness, seq))
                continue

            next_words = [x for x in words_begin if x > pos]
            next_word_start = next_words[0] if next_words else len(transcript_string)
            max_branches = next_word_start - pos

            syl_extensions = range(min_branches, max_branches + 1)

            max_blobs = min(num_blobs_lookahead, len(blob_lengths) - blob_pos)
            blob_extensions = range(1, max_blobs + 1)

            branches += [seq + [i] for i in iter.product(syl_extensions, blob_extensions)]

        print('num branches', len(branches))
        branches += other_seqs
        # filtering step: when two sequences have met the same point (same blob, same syllable), remove the ones with highest cost since they couldn't possibly do any better

        sums_and_seqs = [(current_position_of_seq(x), x) for x in branches]
        sums_and_seqs.sort(key=lambda x: x[0])

        for key, group in iter.groupby(sums_and_seqs, lambda x: x[0]):
            group = [x[1] for x in group]
            scores = [(alignment_fitness(x, blob_lengths, syl_lengths, gap_sizes), x) for x in group]
            best_group_member = min(scores, key=lambda x: x[0])
            new_sequences.append(best_group_member)

        # after previous for loop new_sequences still has a cost associated with each sequence
        # remove all but the least costly sequences
        print('len new sequences', len(new_sequences))
        print('finished seqs', len(finished_seqs))
        new_sequences.sort(key=lambda x: x[0])

        sequences = [x[1] for x in new_sequences][:max_blob_sequences]

        print("----")

    finished_seqs.sort(key=lambda x: x[0])

    res = []
    syl_pos = 0
    blob_pos = 0
    for i, x in enumerate(finished_seqs[0][1]):
        end_syl_index = syl_pos + x[0]
        used_syllables_indices = list(range(syl_pos, end_syl_index))
        syl_pos = end_syl_index
        used_syllables = '-'.join([transcript_string[j] for j in used_syllables_indices])
        syl_length_sum = sum([syl_lengths[j] for j in used_syllables_indices])

        end_blob_index = blob_pos + x[1]
        used_blob_indices = list(range(blob_pos, end_blob_index))
        blob_pos = end_blob_index
        used_blobs = '-'.join([str(blob_lengths[j]) for j in used_blob_indices])
        res.append((used_blobs, used_syllables, syl_length_sum, x))
    print(res)

    draw_blob_alignment(finished_seqs[0][1], transcript_string, cc_groups,
                        image, fname="testimg " + filename + ".png")

    # draw lines representing cc_lines groupings
    col_image = image.image_copy()
    if staff_image:
        col_image = col_image.add_images(staff_image)
    col_image = col_image.to_rgb()
    for line in cc_lines:
        red = np.random.randint(0, 255)
        grn = np.random.randint(0, 255)
        blu = np.random.randint(0, 255)
        for cc in line:
            col_image.draw_hollow_rect(cc.ul, cc.lr, gc.RGBPixel(red, grn, blu), 5)

    # draw lines representing horizontal projection peaks
    imsv(draw_lines(col_image, lines_peak_locs), fname="testimg " + filename + " hlines.png")

# if not single:
#     nums = list(range(11, 21)) + list(range(24, 35))
#     fnames = ['CF-0' + str(x) + '_3' for x in nums]
#     for fn in fnames:
#         process(fn)
