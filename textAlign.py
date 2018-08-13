import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import itertools as iter
import os
import re
import latinSyllabification
from os.path import isfile, join
import numpy as np
import PIL as pil  # python imaging library, for testing only
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
reload(latinSyllabification)

filename = 'CF-011_3'

# PARAMETERS FOR PREPROCESSING
saturation_thresh = 0.6
sat_area_thresh = 150
despeckle_amt = 100            # an int in [1,100]: ignore ccs with area smaller than this
noise_area_thresh = 600        # an int in : ignore ccs with area smaller than this

# PARAMETERS FOR TEXT LINE SEGMENTATION
filter_size = 20                # size of moving-average filter used to smooth projection
prominence_tolerance = 0.50     # log-projection peaks must be at least this prominent
collision_strip_size = 50       # in [0,inf]; amt of each cc to consider when clipping
remove_capitals_scale = 2

# CC GROUPING (BLOBS)
cc_group_gap_min = 16  # any gap at least this wide will be assumed to be a space between words!

letter_width_dict = {
    '*': 20,
    'm': 128,
    'l': 36,
    'i': 36,
    'a': 84,
    'c': 60,
    'e': 59,
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


def imsv(img, fname=''):
    if type(img) == list:
        union_images(img).save_image("testimg " + fname + ".png")
    else:
        img.save_image("testimg " + fname + ".png")


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
                        cc_groups, gamera_image, size=55, fname='testimg.png'):
    '''
    visualizes the alignment given in @alignment_groups.
    '''
    gamera_image.save_image(fname)
    image = pil.Image.open(fname)
    draw = pil.ImageDraw.Draw(image)
    font = pil.ImageFont.truetype('Arial.ttf', size=size)

    cur_syl_index = 0
    for i in range(len(alignment_groups)):
        end_syl_index = cur_syl_index + alignment_groups[i]
        used_syllables_indices = list(range(cur_syl_index, end_syl_index))
        cur_syl_index = end_syl_index
        used_syllables = '-'.join([transcript_string[j] for j in used_syllables_indices])
        used_syllables = str(alignment_groups[i]) + " " + used_syllables

        ul, lr = bounding_box(cc_groups[i])
        position = (ul.x, ul.y - size)
        draw.text(position, used_syllables, fill='rgb(0, 0, 0)', font=font)

        draw.rectangle((ul.x, ul.y, lr.x, lr.y), outline='rgb(0, 0, 0)')

    image.save(fname)


def vertically_coincide(hline_position, comp_offset, comp_nrows, collision=collision_strip_size):
    """
    A helper function that takes in the vertical width of a horizontal strip
    and the vertical measurements of a connected component, and returns a value
    of True if any part of it lies within the strip.
    """

    component_top = comp_offset
    component_bottom = comp_offset + comp_nrows

    strip_top = hline_position - int(collision / 2)
    strip_bottom = hline_position + int(collision / 2)

    both_above = component_top < strip_top and component_bottom < strip_top
    both_below = component_top > strip_bottom and component_bottom > strip_bottom

    return (not both_above and not both_below)


def calculate_peak_prominence(data, index):
    '''
    returns the log of the prominence of the peak at a given index in a given dataset. peak
    prominence gives high values to relatively isolated peaks and low values to peaks that are
    in the "foothills" of large peaks.
    '''
    current_peak = data[index]

    # ignore values at either end of the dataset or values that are not local maxima
    if (index == 0 or
            index == len(data) - 1 or
            data[index - 1] > current_peak or
            data[index + 1] > current_peak or
            (data[index - 1] == current_peak and data[index + 1] == current_peak)):
        return 0

    # by definition, the prominence of the highest value in a dataset is equal to the value itself
    if current_peak == max(data):
        return np.log(current_peak)

    # find index of nearest maxima which is higher than the current peak
    higher_peaks_inds = [i for i, x in enumerate(data) if x > current_peak]

    right_peaks = [x for x in higher_peaks_inds if x > index]
    if right_peaks:
        closest_right_ind = min(right_peaks)
    else:
        closest_right_ind = np.inf

    left_peaks = [x for x in higher_peaks_inds if x < index]
    if left_peaks:
        closest_left_ind = max(left_peaks)
    else:
        closest_left_ind = -np.inf

    right_distance = closest_right_ind - index
    left_distance = index - closest_left_ind

    if (right_distance) > (left_distance):
        closest = closest_left_ind
    else:
        closest = closest_right_ind

    # find the value at the lowest point between the nearest higher peak (the key col)
    lo = min(closest, index)
    hi = max(closest, index)
    between_slice = data[lo:hi]
    key_col = min(between_slice)

    prominence = np.log(data[index] - key_col + 1)

    return prominence


def find_peak_locations(data, tol=prominence_tolerance):
    prominences = [(i, calculate_peak_prominence(data, i)) for i in range(len(data))]

    # normalize to interval [0,1]
    prom_max = max([x[1] for x in prominences])
    prominences[:] = [(x[0], x[1] / prom_max) for x in prominences]

    # take only the tallest peaks above given tolerance
    peak_locs = [x for x in prominences if x[1] > tol]

    # if a peak has a flat top, then both 'corners' of that peak will have high prominence; this
    # is rather unavoidable. just check for adjacent peaks with exactly the same prominence and
    # remove the lower one
    to_remove = [peak_locs[i] for i in range(len(peak_locs) - 2)
                if peak_locs[i][1] == peak_locs[i+1][1]]
    for r in to_remove:
        peak_locs.remove(r)

    peak_locs[:] = [x[0] for x in peak_locs]

    return peak_locs


def moving_avg_filter(data, filter_size=filter_size):
    '''
    returns a list containing the data in @data filtered through a moving-average filter of size
    @filter_size to either side; that is, filter_size = 1 gives a size of 3, filter size = 2 gives
    a size of 5, and so on.
    '''
    smoothed = [0] * len(data)
    for n in range(filter_size, len(data) - filter_size):
        vals = data[n - filter_size: n + filter_size + 1]
        smoothed[n] = np.mean(vals)
    return smoothed


def preprocess_images(input_image, staff_image,
                    sat_tresh=saturation_thresh, sat_area_thresh=sat_area_thresh,
                    despeckle_amt=despeckle_amt, filter_runs=10):

    image_sats = input_image.saturation().to_greyscale().threshold(int(saturation_thresh * 256))
    image_bin = input_image.to_onebit().subtract_images(image_sats)
    staff_image = staff_image.to_onebit()

    # keep only colored ccs above a certain size
    ccs = image_bin.cc_analysis()
    for c in ccs:
        area = c.nrows
        if sat_area_thresh < area:
            c.fill_white()

    image_bin = input_image.to_onebit().subtract_images(image_bin)
    image_bin.invert()
    image_bin.despeckle(despeckle_amt)
    image_bin.invert()
    image_bin.reset_onebit_image()

    # find likely rotation angle and correct
    angle, tmp = image_bin.rotation_angle_projections()
    image_bin = image_bin.rotate(angle=angle)
    staff_image = staff_image.rotate(angle=angle)
    for i in range(filter_runs):
        image_bin.filter_short_runs(5, 'black')
        image_bin.filter_narrow_runs(5, 'black')
        staff_image.filter_narrow_runs(50, 'white')
    staff_image.despeckle(despeckle_amt)

    return image_bin, staff_image


def identify_text_lines(image_bin):

    # compute y-axis projection of input image and filter with sliding window average
    print('finding projection peaks...')
    project = image_bin.projection_rows()
    smoothed_projection = moving_avg_filter(project, filter_size)

    # calculate normalized log prominence of all peaks in projection
    peak_locations = find_peak_locations(smoothed_projection)

    # perform connected component analysis and remove sufficiently small ccs and ccs that are too
    # tall; assume these to be ornamental letters
    print('connected component analysis...')
    components = image_bin.cc_analysis()

    for c in components:
        if c.black_area()[0] < noise_area_thresh:
            c.fill_white()

    components[:] = [c for c in components if c.black_area()[0] > noise_area_thresh]

    med_comp_height = np.median([c.nrows for c in components])

    components[:] = [c for c in components if c.nrows < (med_comp_height * remove_capitals_scale)]

    # using the peak locations found earlier, find all connected components that are intersected by
    # a horizontal strip at either edge of each line. these are the lines of text in the manuscript
    print('intersecting connected components with text lines...')
    cc_lines = []
    for line_loc in peak_locations:
        res = [x for x in components if vertically_coincide(line_loc, x.offset_y, x.nrows)]
        res = sorted(res, key=lambda x: x.offset_x)
        cc_lines.append(res)

    # if a single connected component appears in more than one cc_line, give priority to the line
    # that is closer to the center of the component's bounding box
    # TODO: SOMETHING IS GOING VERY WRONG HERE????? LOOK AT FOLIO 12
    for n in range(len(cc_lines) - 1):
        intersect = set(cc_lines[n]) & set(cc_lines[n + 1])

        # if most of the ccs are shared between these lines, just delete one of them
        if len(intersect) > (0.5 * min(len(cc_lines[n]), len(cc_lines[n + 1]))):
            cc_lines[n] = []
            continue

        for i in intersect:

            box_center = i.offset_y + (i.nrows / 2)
            distance_up = abs(peak_locations[n] - box_center)
            distance_down = abs(peak_locations[n + 1] - box_center)

            if distance_up > distance_down:
                cc_lines[n].remove(i)
                # print('removing up')
            else:
                cc_lines[n+1].remove(i)
                # print('removing down')

    # remove all empty lines from cc_lines in case they've been created by previous steps
    cc_lines[:] = [x for x in cc_lines if bool(x)]

    return cc_lines, peak_locations


def find_ccs_under_staves(cc_lines, staff_image,
            min_line_length_scale=0.2, max_distance_to_staff=199):
    '''
    actual musical text must have a staff immediately above it and should NOT be on the same horizontal position as any staves. this function checks every connected component in cc_lines and removes those that do not meet these criteria
    '''

    proj = moving_avg_filter(staff_image.projection_rows())
    staff_peaks = find_peak_locations(proj)

    cc_lines_flat = reduce((lambda x, y: x + y), cc_lines)
    staff_ccs = staff_image.cc_analysis()
    longest_line = max([x.ncols for x in staff_ccs])
    # staff_ccs[:] = [x for x in staff_ccs if x.ncols > min_line_length_scale * longest_line]

    # first filter out all ccs not directly below a staff
    # for every cc, bring a line down across the whole page thru its center.
    for i in reversed(range(len(cc_lines))):

        to_remove = set()

        for j in range(len(cc_lines[i])):
            cur_comp = cc_lines[i][j]
            comp_center_h = cur_comp.offset_x + (cur_comp.ncols / 2)

            # all staff lines that cross this vertical line, above or below
            lines_cross = [x for x in staff_ccs if
                x.offset_x <= comp_center_h <= x.offset_x + x.ncols]

            # all other components that cross this vertical line, above or below
            ccs_cross = [x for x in cc_lines_flat if
                x.offset_x <= comp_center_h <= x.offset_x + x.ncols]

            # find vertical position of first staff line above this component
            lines_cross_above = [x.offset_y for x in lines_cross if
                x.offset_y < cur_comp.offset_y]
            closest_line_pos = max(lines_cross_above + [0])

            # find vertical position of first other component above this component
            lines_cross_above = [x.offset_y for x in ccs_cross if
                x.offset_y < cur_comp.offset_y]
            closest_cc_pos = max(lines_cross_above + [0])

            # print(closest_cc_pos, closest_line_pos, cur_comp.offset_y)
            distance_to_staff = cur_comp.offset_y - closest_line_pos

            if (closest_cc_pos > closest_line_pos
                    or closest_line_pos == 0
                    or distance_to_staff > max_distance_to_staff):
                to_remove.add(cur_comp)

            # next, remove everything horizontally aligned with a staff

            comp_center_v = cur_comp.offset_y + (cur_comp.nrows / 2)
            lines_cross_h = min([abs(x - comp_center_v) for x in staff_peaks])

            if lines_cross_h > max_distance_to_staff:
                to_remove.add(cur_comp)

        for tr in to_remove:
            cc_lines[i].remove(tr)

    cc_lines[:] = [x for x in cc_lines if bool(x)]

    return cc_lines


def group_ccs(cc_list, gap_tolerance=cc_group_gap_min):
    '''
    a helper function that takes in a list of ccs on the same line and groups them together based
    on contiguity of their bounding boxes along the horizontal axis.
    '''

    cc_copy = cc_list[:]
    result = [[cc_copy.pop(0)]]

    # iterate over copy of this the line, removing
    while(cc_copy):

        current_group = result[-1]
        left_bound = min([x.offset_x for x in current_group]) - gap_tolerance
        right_bound = max([x.offset_x + x.ncols for x in current_group]) + gap_tolerance

        overlaps = [x for x in cc_copy if
                    (left_bound <= x.offset_x <= right_bound) or
                    (left_bound <= x.offset_x + x.ncols <= right_bound)
                    ]

        if not overlaps:
            result.append([cc_copy.pop(0)])
            continue

        for x in overlaps:
            result[-1].append(x)
            cc_copy.remove(x)

    gap_sizes = []
    for n in range(len(result)-1):
        left = result[n][-1].offset_x + result[n][-1].ncols
        right = result[n+1][0].offset_x
        gap_sizes.append(right - left)

    return result  # , gap_sizes


def parse_transcript(filename, syllables=False):
    file = open(filename, 'r')
    lines = file.readlines()
    lines = [x for x in lines if not x[0] == '#']
    lines = ['*' + x[1:] for x in lines]
    lines = ' '.join(lines)
    file.close()

    lines = lines.lower()
    words_begin = []

    if not syllables:
        lines = lines.replace('-', '')
        lines = lines.replace(' ', '')
        lines = lines.replace('\n', '')
    else:
        lines = lines.replace('.', '')
        lines = lines.replace(' ', '- ')
        lines = lines.replace('\n', '')
        lines = re.compile('[-]').split(lines)
        words_begin.append(0)
        for i, x in enumerate(lines):
            if x[0] == ' ':
                lines[i] = lines[i][1:]
                words_begin.append(i)

    return lines, words_begin


def alignment_fitness(alignment, group_lengths, syl_lengths):
    '''
    given an alignment between text blobs on the manuscript and syllables of the transcript,
    computes the error of the alignment based on the estimated syllable lengths and the known
    lengths present in the original image.
    '''
    cur_pos = 0
    cost = 0
    for i, x in enumerate(alignment):
        new_sum = sum(syl_lengths[cur_pos:cur_pos + x])
        cur_pos += x
        cost += abs(new_sum - group_lengths[i]) ** 2

    return round(cost / len(alignment))


# if __name__ == "__main__":
def process(filename):
    print('processing ' + filename + '...')

    raw_image = gc.load_image('./png/' + filename + '.png')
    staff_image = gc.load_image('./png/' + filename[:-1] + '2.png')
    image, staff_image = preprocess_images(raw_image, staff_image)
    cc_lines, lines_peak_locs = identify_text_lines(image)

    cc_lines = find_ccs_under_staves(cc_lines, staff_image)

    cc_groups = []
    for x in cc_lines:
        cc_groups += group_ccs(x)

    group_lengths = []
    for g in cc_groups:
        width = sum([x.ncols for x in g])
        group_lengths.append(width)

    transcript_string, words_begin = latinSyllabification.parse_transcript(
            './png/' + filename + '.txt')
    transcript_lengths = [len(x) for x in transcript_string]

    # estimate length of each syllable
    avg_char_length = round(sum(group_lengths) / sum(transcript_lengths))
    avg_syl_length = round(sum(group_lengths) / len(group_lengths))
    syl_lengths = []
    letter_dict = defaultdict(lambda: avg_char_length, **letter_width_dict)
    for syl in transcript_string:
        this_width = 0
        for char in syl:
            this_width += letter_dict[char]
        syl_lengths.append(this_width)

    # set up for sequence searching
    sequences = [[]]
    max_blob_sequences = 5000  # probably unnecessary but just in case, so nothing gets stuck

    # iterating over blobs
    for i, gl in enumerate(group_lengths):
        print('sequences begin length', len(sequences))
        print(sequences[0])

        new_sequences = []
        print('group length', gl)

        branches = []
        for seq in sequences:

            # lower bound on number of syllables that could possibly be assigned to this blob?
            min_branches = 0

            # max number of branches = branch syllables until reach end of current word
            pos = sum(seq)

            if pos == len(transcript_string):
                branches.append(seq)
                continue

            next_words = [x for x in words_begin if x > pos]
            next_word_start = next_words[0] if next_words else len(transcript_string)
            max_branches = next_word_start - pos + 1

            branches += [seq + [i] for i in range(min_branches, max_branches)]

        print('num branches', len(branches))
        sums_and_seqs = [(sum(x), x) for x in branches]
        sums_and_seqs.sort(key=lambda x: x[0])

        for key, group in iter.groupby(sums_and_seqs, lambda x: x[0]):
            group = [x[1] for x in group]
            scores = [(alignment_fitness(x, group_lengths, syl_lengths), x) for x in group]
            best_group_member = min(scores, key=lambda x: x[0])
            new_sequences.append(best_group_member)

        # after previous for loop new_sequences still has a cost associated with each sequence
        # remove all but the least costly sequences
        print('len new sequences', len(new_sequences))
        new_sequences.sort(key=lambda x: x[0])

        sequences = [x[1] for x in new_sequences][:max_blob_sequences]

        print("----")

    draw_blob_alignment(sequences[0], transcript_string, cc_groups,
                        image, fname="testimg " + filename + ".png")

    res = []
    pos = 0
    for i, x in enumerate(sequences[0]):
        end_syl_index = pos + x
        used_syllables_indices = list(range(pos, end_syl_index))
        pos = end_syl_index
        used_syllables = '-'.join([transcript_string[j] for j in used_syllables_indices])
        syl_length_sum = sum([syl_lengths[j] for j in used_syllables_indices])
        res.append((group_lengths[i], syl_length_sum, used_syllables,  x))


nums = list(range(11, 21)) + list(range(24, 35))
fnames = ['CF-0' + str(x) + '_3' for x in nums]
for fn in fnames:
    process(fn)


# proj = moving_avg_filter(staff_image.projection_rows())
# staff_peaks = find_peak_locations(proj)
# imsv(draw_lines(image, staff_peaks))
# plot([np.log(x+1) / np.log(max(proj)) for x in proj])
