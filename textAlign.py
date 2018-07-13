import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import os
import re
import syllable
from os.path import isfile, join
import numpy as np
reload(syllable)

filename = 'CF-019_3.png'
despeckle_amt = 100             # an int in [1,100]: ignore ccs with area smaller than this
noise_area_thresh = 500        # an int in : ignore ccs with area smaller than this

filter_size = 20                # size of moving-average filter used to smooth projection
prominence_tolerance = 0.90      # y-axis projection peaks must be at least this prominent

collision_strip_size = 50       # in [0,inf]; amt of each cc to consider when clipping
horizontal_gap_tolerance = 20   # value in pixels


def _bases_coincide(hline_position, comp_offset, comp_nrows, collision=collision_strip_size):
    """
    A helper function that takes in the vertical width of a horizontal strip
    and the vertical measurements of a connected component, and returns a value
    of True if the bottom of the connected component lies within the strip.

    If the connected component is shorter than the height of the strip, then
    we just check if any part of it lies within the strip at all.
    """

    component_top = comp_offset
    component_bottom = comp_offset + comp_nrows

    strip_top = hline_position - int(collision / 2)
    strip_bottom = hline_position + int(collision / 2)

    both_above = component_top < strip_top and component_bottom < strip_top
    both_below = component_top > strip_bottom and component_bottom > strip_bottom

    return (not both_above and not both_below)


def _group_ccs(cc_list, gap_tolerance=horizontal_gap_tolerance):
    '''
    a helper function that takes in a list of ccs on the same line and groups them together based
    on contiguity of their bounding boxes along the horizontal axis.
    '''

    cc_copy = cc_list[:]
    result = [[cc_copy.pop(0)]]

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

    return result, gap_sizes


def _exhaustively_bunch(cc_list, gap_tolerance=horizontal_gap_tolerance, max_num_ccs=7):
    '''
    given a list of connected components on a single line (assumed to be in order from left
    to right), groups them into all possible bunches of up to max_num_ccs consecutive
    components. bunches with gaps larger than horizontal_gap_tolerance are not added.
    '''

    cc_copy = cc_list[:]
    result = []

    for n in range(1, max_num_ccs):
        next_group = [cc_copy[x:x + n] for x in range(len(cc_copy) - n + 1)]

        # for each member of this group, decide if it should be added; make sure the connected
        # components are separated by less than horizontal_gap_tolerance

        for g in next_group:
            cc_end = [x.offset_x + x.ncols for x in g[:-1]]
            cc_begin = [x.offset_x for x in g[1:]]
            gaps = [cc_begin[x] - cc_end[x] for x in range(n-1)]

            if not any([x >= horizontal_gap_tolerance for x in gaps]):
                result.append(g)

    # result += [[x] for x in cc_copy]
    return result


def _bounding_box(cc_list):
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


def _calculate_peak_prominence(data, index):
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


def _find_peak_locations(data, tol=prominence_tolerance):
    prominences = [(i, _calculate_peak_prominence(data, i))
                   for i in range(len(data))]
    prom_max = max([x[1] for x in prominences])
    prominences[:] = [(x[0], x[1] / prom_max) for x in prominences]
    peak_locs = [x[0] for x in prominences if x[1] > tol]
    return peak_locs


def _moving_avg_filter(data, filter_size):
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


def _identify_text_lines_and_bunch(input_image):
    # find likely rotation angle and correct
    print('correcting rotation...')
    image_grey = input_image.to_greyscale()
    image_bin = image_grey.to_onebit()
    angle, tmp = image_bin.rotation_angle_projections()
    image_bin = image_bin.rotate(angle=angle)
    # image_bin = image_bin.erode_dilate(2,1,1)
    image_bin.filter_short_runs(3, 'black')
    image_bin.filter_narrow_runs(3, 'black')
    image_bin.despeckle(despeckle_amt)

    # compute y-axis projection of input image and filter with sliding window average
    print('finding projection peaks...')
    project = image_bin.projection_rows()
    smoothed_projection = _moving_avg_filter(project, filter_size)

    # calculate normalized log prominence of all peaks in projection
    peak_locations = _find_peak_locations(smoothed_projection)

    # perform connected component analysis and remove sufficiently small ccs and ccs that are too
    # tall; assume these to be ornamental letters
    print('connected component analysis...')
    components = image_bin.cc_analysis()
    med_comp_height = np.median([c.nrows for c in components])
    components[:] = [c for c in components if c.black_area()[0] > noise_area_thresh and c.nrows < (med_comp_height * 2)]

    # using the peak locations found earlier, find all connected components that are intersected by a
    # horizontal strip at either edge of each line. these are the lines of text in the manuscript
    print('intersecting connected components with text lines...')
    cc_lines = []
    for line_loc in peak_locations:
        res = [x for x in components if _bases_coincide(line_loc, x.offset_y, x.nrows)]
        res = sorted(res, key=lambda x: x.offset_x)
        cc_lines.append(res)

    # if a single connected component appears in more than one cc_line, give priority to the line
    # that is closer to the cemter of the component's bounding box
    for n in range(len(cc_lines)-1):
        intersect = set(cc_lines[n]) & set(cc_lines[n+1])

        for i in intersect:
            box_center = i.offset_y + (i.nrows / 2)
            distance_up = abs(peak_locations[n] - box_center)
            distance_down = abs(peak_locations[n+1] - box_center)

            if distance_up < distance_down:
                cc_lines[n].remove(i)
            else:
                cc_lines[n+1].remove(i)

    # remove all empty lines from cc_lines in case they've been created by previous steps
    cc_lines[:] = [x for x in cc_lines if bool(x)]

    syllable_list = []

    # now perform oversegmentation on each line: get the bounding box around each line, get the
    # horizontal projection for what's inside that bounding box, filter it, find the peaks using
    # log-prominence, draw new bounding boxes around individual letters / parts of letters using
    # the peaks locations found, and finally compute an exhaustive bunching of these boxes
    print('oversegmenting and bunching...')
    for line_num, cl in enumerate(cc_lines):

        ul, lr = _bounding_box(cl)
        line_image = image_bin.subimage(ul, lr)
        line_proj = line_image.projection_cols()
        line_proj = [max(line_proj) - x for x in line_proj]  # reverse it
        char_filter_size = 5
        letter_horizontal_tolerance = 3
        smooth_line_proj = _moving_avg_filter(line_proj, char_filter_size)

        peak_locs = _find_peak_locations(smooth_line_proj)
        peak_locs = [x for i, x in enumerate(peak_locs)
                     if (i == 0) or (x - peak_locs[i-1] > letter_horizontal_tolerance)]

        boxes = []
        for n in range(len(peak_locs) - 1):

            # if everything between these peaks is empty, then skip it
            if all([x == max(smooth_line_proj) for x in smooth_line_proj[peak_locs[n]:peak_locs[n+1]]]):
                continue

            ul = gc.Point(line_image.offset_x + peak_locs[n], line_image.offset_y)
            lr = gc.Point(
                line_image.offset_x + peak_locs[n+1],
                line_image.offset_y + line_image.nrows)
            boxes.append(image_bin.subimage(ul, lr).trim_image())

        bunches = _exhaustively_bunch(boxes)
        bunches.sort(key=lambda x: min(y.offset_x for y in x))

        for b in bunches:
            bunch_image = union_images(b)
            new_syl = syllable.Syllable(image=bunch_image)
            new_syl.box_index = min(y.offset_x for y in b) + (image_bin.ncols * line_num)
            syllable_list.append(new_syl)

    return {'image': image_bin,
            'sliced': syllable_list,
            'peaks': peak_locations,
            'cc_lines': cc_lines,
            'projection': smoothed_projection}


def _parse_transcript_syllables(filename):

    res = []

    file = open(filename, 'r')
    lines = (file.readlines())
    file.close()

    for l in lines:
        phrase = re.split('[\s-]', l)
        res += [syllable.Syllable(text=x.lower()) for x in phrase if x]

    return res

# LOCAL HELPER FUNCTIONS - DON'T END UP IN RODAN


def imsv(img, fname=''):
    if type(img) == list:
        union_images(img).save_image("testimg " + fname + ".png")
    elif type(img) == syllable.Syllable:
        img.image.save_image("testimg " + fname + ".png")
        print(img.text)
    else:
        img.save_image("testimg " + fname + ".png")


def draw_syl_boxes_imsv(img, syl):
    new_img = union_images([img.image_copy(), syl.image])
    new_img.draw_hollow_rect(syl.ul, syl.lr, 1, 9)
    for ps in syl.predicted_syllable:
        new_img.draw_hollow_rect(ps.ul, ps.lr, 1, 9)
    imsv(new_img)


def two_boxes(img, tran_syls, pred_syls, index):
    new_img = union_images([img.image_copy(), tran_syls[index].image])
    new_img.draw_hollow_rect(tran_syls[index].ul, tran_syls[index].lr, 1, 6)
    new_img.draw_hollow_rect(pred_syls[index][0].ul, pred_syls[index][0].lr, 1, 6)
    imsv(new_img)


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
        new.draw_line(start, end, 1, 5)
    return new


if __name__ == "__main__":

    # filenames = os.listdir('./png')
    # filenames = ['CF-011_3']
    # for fn in filenames:
    fn = 'CF-011_3'

    print('processing ' + fn + '...')

    image = gc.load_image('./png/' + fn + '.png')
    processed_image = _identify_text_lines_and_bunch(image)
    image = processed_image['image']
    manuscript_syls = processed_image['sliced']
    peak_locs = processed_image['peaks']
    cc_lines = processed_image['cc_lines']

    transcript_syls = _parse_transcript_syllables('./png/' + fn + '.txt')

    # normalize extracted features
    all_syls = manuscript_syls + transcript_syls

    for fk in all_syls[0].features.keys():
        avg = np.mean([x.features[fk] for x in all_syls])
        std = np.std([x.features[fk] for x in all_syls])
        for n in range(len(manuscript_syls)):
            manuscript_syls[n].features[fk] = (manuscript_syls[n].features[fk] - avg) / std
        for n in range(len(transcript_syls)):
            transcript_syls[n].features[fk] = (transcript_syls[n].features[fk] - avg) / std

    print('performing comparisons...')
    seq_arr = []
    for ts in transcript_syls:
        res = syllable.knn_search(manuscript_syls, ts, 1200)
        # found_syl = min(res, key = lambda x: x[1])[0]
        seq_arr.append(res)

    # find diagonal path that is strictly increasing with minimum total weight
    # s is list of candidates for the current syllable
    print('finding optimal alignment...')
    identified_syls = []
    current_position = 0
    for s in seq_arr:
        following_syls = [x for x in s if x[0].box_index >= current_position]
        next_syl = min(following_syls, key=lambda x: x[0].box_index)
        identified_syls.append(next_syl)
        current_position = next_syl[0].box_index + next_syl[0].ncols

    # hopefully, this will reach the end of the manuscript! if not, then the transcript definitely
    # has material that extends past the end of the transcript
    current_position = 0
    criteria_met = False
    while not criteria_met:

        replacements = []

        # headroom is the space each syllable has to itself horizontally, a range
        for i, current_syl in enumerate(identified_syls):

            cur_candidates = seq_arr[i]

            if i == 0:
                left = 0
            else:
                prev = identified_syls[i - 1][0]
                left = prev.box_index + prev.ncols

            if i == len(identified_syls) - 1:
                right == np.inf
            else:
                right = identified_syls[i + 1][0].box_index

            headroom = (left, right)

            # find just the candidates that are inside of the headroom range

            clip_candidates = [x for x in cur_candidates
                if left <= x[0].box_index <= right
                and left <= x[0].box_index + x[0].ncols <= right
                and x[0] != current_syl]

            if not bool(clip_candidates):
                replacements.append((None, -1))
            else:
                best_candidate = min(clip_candidates, key=lambda x: x[1])
                difference = current_syl[1] - best_candidate[1]
                replacements.append((best_candidate, difference))

        if all([x[1] <= 0 for x in replacements]):
            criteria_met = True
            continue

        # find the section that can be expanded yielding the largest reduction in weight
        best_replacement_ind = max(range(len(replacements)),
            key=lambda x: replacements[x][1])
        best_replacement = replacements[best_replacement_ind][0]
        print(replacements[best_replacement_ind])
        identified_syls[best_replacement_ind] = best_replacement

    # for ids in [x[0] for x in identified_syls]:
    #     image.draw_hollow_rect(ids.ul,ids.lr,1,5)
    # imsv(image)


# temp = draw_lines(image,[x + line_image.offset_x for x in peak_locs],False)
# imsv(temp)

# for b in bunched:
#    ub = union_images(b)
#    image.draw_hollow_rect(ub.ul,ub.lr,1,5)
# imsv(image)

# plt.clf()
# plt.scatter([x[0] for x in prominences], [x[1] for x in prominences], s = 5)
# plt.plot([x / max(smoothed_proj) for x in smoothed_proj], linewidth=1, color='k')
# plt.savefig("testplot " + filename + ".png", dpi=800)
