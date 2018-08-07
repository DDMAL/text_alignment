import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
import networkx as nx
import itertools as iter
import os
import re
import textUnit
from os.path import isfile, join
import numpy as np
import PIL as pil  # python imaging library, for testing only
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
reload(textUnit)

filename = 'CF-017_3'

# PARAMETERS FOR PREPROCESSING
saturation_thresh = 0.7
sat_area_thresh = 150
despeckle_amt = 100            # an int in [1,100]: ignore ccs with area smaller than this
noise_area_thresh = 1200        # an int in : ignore ccs with area smaller than this

# PARAMETERS FOR TEXT LINE SEGMENTATION
filter_size = 20                # size of moving-average filter used to smooth projection
prominence_tolerance = 0.80     # log-projection peaks must be at least this prominent
collision_strip_size = 50       # in [0,inf]; amt of each cc to consider when clipping
remove_capitals_scale = 2

char_filter_size = 5

# CC GROUPING (BLOBS)
letter_horizontal_tolerance = 10
cc_group_gap_min = 14  # any gap at least this wide will be assumed to be a space between words!
min_letter_width = 35
max_noise_width = 50

letter_width_dict = {
    '*': 20,
    # 'm': 128,
    # 'l': 36,
    # 'i': 36,
    # 'a': 84,
    # 'c': 60,
    # 'e': 59,
    # 'r': 60
}

# PARAMETERS FOR GRAPH SEARCH
max_num_ccs = 5
max_num_sequences = 400


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
    elif type(img) == textUnit.textUnit:
        img.image.save_image("testimg " + fname + ".png")
        print(img.text)
    else:
        img.save_image("testimg " + fname + ".png")


def seq_boxes_imsv(img, sequence, graph):
    img_copy = img.image_copy()

    nodes = sequence.seq
    for i in range(len(nodes) - 1):
        unit = graph[nodes[i]][nodes[i+1]]['object']
        img_copy.draw_hollow_rect(unit.ul, unit.lr, 1, 9)
    imsv(img_copy)


def draw_seq_boxes_imsv(img, prototypes, graph, seq, index):
    new_img = union_images([img.image_copy(), prototypes[seq.predicted_string[index][0]].image])
    unit = graph[seq.seq[index]][seq.seq[index+1]]['object']
    new_img.draw_hollow_rect(unit.ul, unit.lr, 1, 9)
    print(seq.predicted_string[index])
    imsv(new_img)
    # draw_seq_boxes_imsv(image,prototypes,graph,completed_sequences[0],0)


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


def bases_coincide(hline_position, comp_offset, comp_nrows, collision=collision_strip_size):
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


def moving_avg_filter(data, filter_size):
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


def preprocess_image(input_image, sat_tresh=saturation_thresh, sat_area_thresh=sat_area_thresh,
            despeckle_amt=despeckle_amt, filter_runs=30):

    image_sats = input_image.saturation().to_greyscale().threshold(int(saturation_thresh * 256))
    image_bin = input_image.to_onebit().subtract_images(image_sats)

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
    for i in range(filter_runs):
        image_bin.filter_short_runs(5, 'black')
        image_bin.filter_narrow_runs(5, 'black')
    image_bin.despeckle(despeckle_amt)

    return image_bin


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
    med_comp_height = np.median([c.nrows for c in components])

    for c in components:
        if c.black_area()[0] < noise_area_thresh:
            c.fill_white()

    components[:] = [c for c in components if c.black_area()[0] > noise_area_thresh]
    components[:] = [c for c in components if c.nrows < (med_comp_height * remove_capitals_scale)]

    # using the peak locations found earlier, find all connected components that are intersected by
    # a horizontal strip at either edge of each line. these are the lines of text in the manuscript
    print('intersecting connected components with text lines...')
    cc_lines = []
    for line_loc in peak_locations:
        res = [x for x in components if bases_coincide(line_loc, x.offset_y, x.nrows)]
        res = sorted(res, key=lambda x: x.offset_x)
        cc_lines.append(res)

    # if a single connected component appears in more than one cc_line, give priority to the line
    # that is closer to the center of the component's bounding box
    # TODO: SOMETHING IS GOING VERY WRONG HERE????? LOOK AT FOLIO 12
    for n in range(len(cc_lines) - 1):
        intersect = set(cc_lines[n]) & set(cc_lines[n + 1])

        print(len(intersect), len(cc_lines[n]), len(cc_lines[n + 1]))
        # if most of the ccs are shared between these lines, just delete one of them
        if len(intersect) > (0.5 * min(len(cc_lines[n]), len(cc_lines[n + 1]))):
            print('removing')
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


def segment_lines_build_graph(image_bin, cc_lines):

    print('oversegmenting and building initial graph...')

    syllable_list = []
    graph = nx.DiGraph()
    previous_node = (0, 0)
    all_nodes = []
    all_nodes.append(previous_node)
    graph.add_node(previous_node)

    # now perform oversegmentation on each line: get the bounding box around each line, get the
    # horizontal projection for what's inside that bounding box, filter it, find the peaks using
    # log-prominence, draw new bounding boxes around individual letters / parts of letters using
    # the peaks locations found, and create a directed straight-line graph
    all_peak_locs = []
    all_line_images = []
    projections = []

    for line_num, cl in enumerate(cc_lines):

        ul, lr = bounding_box(cl)
        line_image = image_bin.subimage(ul, lr)
        # this is ugly and it'd be much better to have another layer of classified ornamental
        # letters to deal with, but it's what's gotta happen for now.
        # ul.x = 0
        # lr.x = image_bin.ncols - 1
        # temp_image = image_bin.subimage(ul, lr)
        # new_ul = gc.Point(int(min(temp_image.contour_left())), ul.y)
        # new_lr = gc.Point(int(lr.x - min(temp_image.contour_right())), lr.y)
        # line_image = image_bin.subimage(new_ul, new_lr)

        all_line_images.append(line_image)
        line_proj = line_image.projection_cols()
        line_proj = [max(line_proj) - x for x in line_proj]  # reverse it

        smooth_line_proj = moving_avg_filter(line_proj, char_filter_size)

        projections.append(smooth_line_proj)

        peak_locs = find_peak_locations(smooth_line_proj)
        peak_locs.insert(0, 0)
        peak_locs.append(len(smooth_line_proj) - 1)
        peak_locs = [x for i, x in enumerate(peak_locs)
                     if (i == 0)
                     or (x - peak_locs[i-1] > letter_horizontal_tolerance)]

        all_peak_locs.append(peak_locs)

        for n in range(len(peak_locs) - 1):

            # if everything between these peaks is empty, then skip it
            if all([x == max(smooth_line_proj) for x in smooth_line_proj[peak_locs[n]:peak_locs[n+1]]]):
                continue

            ul = gc.Point(line_image.offset_x + peak_locs[n], line_image.offset_y)
            lr = gc.Point(
                line_image.offset_x + peak_locs[n+1],
                line_image.offset_y + line_image.nrows)
            current_box = image_bin.subimage(ul, lr).trim_image()
            next_node = (line_num, lr.x)
            graph.add_edge(previous_node, next_node, object=current_box)
            previous_node = next_node
            all_nodes.append(previous_node)

    edges_to_add = []
    print('adding additional lines to graph...')

    # starting at 3 because edges correspond to image slices and we index through nodes; one node
    # contains no image, and two nodes contain a single image (which is already handled in the
    # previous loop. Three nodes contain two edges between them, and thus two images.
    for current_node, num_nodes in iter.product(all_nodes, range(3, max_num_ccs)):
        group = [current_node]
        group_images = []
        target_node = current_node

        # create list of node identifiers num_nodes long, starting from current_node.
        # add to group_images the image associated with each edge traversed.
        is_group_valid = True
        for i in range(num_nodes - 1):
            adj_node = graph[group[-1]].keys()

            if not adj_node:
                is_group_valid = False
                break

            group.append(adj_node[0])
            prev = group[i]
            next = group[i + 1]
            group_images.append(graph.edges[prev, next]['object'])

        if not is_group_valid:
            continue

        # we only want to add this as an edge if the images are sufficiently close together that
        # it makes sense to consider them as a single character or ligature.
        add_this_group = False

        for i in range(len(group_images) - 1):
            prev_right = group_images[i].offset_x + group_images[i].ncols
            next_left = group_images[i + 1].offset_x
            # also check to make sure distance is greater than -1; if less, then this group
            # occurs across a line break, and it should be discarded
            add_this_group = (-1 <= next_left - prev_right <= letter_horizontal_tolerance)

        if (add_this_group):
            edge_image = union_images(group_images)
            edges_to_add.append((group[0], group[-1], {'object': edge_image}))

    # add to graph all edges deemed valid
    graph.add_edges_from(edges_to_add)

    # transform the content of every edge from an image in to a textUnit containing that image
    for nodes in graph.edges:
        edge_image = graph.edges[nodes[0], nodes[1]]['object']
        unit = textUnit.textUnit(image=edge_image)
        graph.edges[nodes[0], nodes[1]]['object'] = unit

    for node in graph.nodes:
        graph.nodes[node]['leaderboard'] = {}

    return graph, all_peak_locs, all_line_images, projections


def parse_transcript(filename, syllables=False):
    file = open(filename, 'r')
    lines = file.readlines()
    lines = ['*' + x[1:] for x in lines]
    lines = ' '.join([x for x in lines if not x[0] == '#'])
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


def next_possible_prototypes(string, prototypes):
    res = {}

    units = prototypes.keys()

    for u in units:
        split_u = u.split('_')[0]
        comp = string[0:len(split_u)]
        if split_u == comp:
            res[u] = prototypes[u]

    return res


def get_branches_of_sequence(current_seq, graph):
    '''
    given a sequence and the graph of slice positions, returns all the branches that could be
    made from the head of that sequence.
    '''
    current_head = current_seq.seq[-1]
    candidates = next_possible_prototypes(transcript_string[current_seq.char_index:], prototypes)
    branches = []

    # if candidates is empty, that means this sequence is at the end of the transcript string.
    if not candidates:
        return True, False

    # if there are no successors but there ARE still candidates available, then this sequence is at # the end of the manuscript(s) input but it has not yet completed its alignment. throw it out!
    if not list(graph.successors(current_head)):
        return False, False

    # successors to the node at the head of the current sequence
    for suc in graph.successors(current_head):

        this_edge_unit = graph[current_head][suc]['object']

        candidate_scores = {}

        # compare the image from each edge to each image in candidates
        # the image that will be chosen is the one with lowest cost
        for c in candidates.keys():
            candidate_scores[c] = textUnit.compare_units(candidates[c], this_edge_unit)
        chosen_candidate_key = min(candidate_scores, key=candidate_scores.get)
        chosen_len = len(chosen_candidate_key.split('_')[0])
        # just making an entirely new object by hand instead of copying; much faster
        new_seq = list(current_seq.seq) + [suc]
        new_used_edges = current_seq.used_edges + [(current_seq.head(), suc)]
        # combine averages
        new_cost_arr = current_seq.cost_arr + [candidate_scores[chosen_candidate_key]] * chosen_len
        new_index = current_seq.char_index + chosen_len
        new_string = current_seq.predicted_string + [[chosen_candidate_key]]

        lead = graph.nodes[suc]['leaderboard']
        mean_cost = np.mean(new_cost_arr)
        if new_index not in lead.keys():
            lead[new_index] = mean_cost
        elif lead[new_index] >= mean_cost:
            lead[new_index] = mean_cost
        else:
            continue

        branches.append(textUnit.unitSequence(
            seq=new_seq,
            used_edges=new_used_edges,
            char_index=new_index,
            cost_arr=new_cost_arr,
            predicted_string=new_string
            ))

    # test to see if leaderboard for heads of branches excludes branches from competition and
    # update leaderboards if not

    return branches, True


def test_text(gamera_image, seq, graph, size=70, fname='testimg.png'):
    gamera_image.save_image(fname)
    image = pil.Image.open(fname)
    draw = pil.ImageDraw.Draw(image)
    font = pil.ImageFont.truetype('Arial.ttf', size=size)
    edges = list(seq.used_edges)

    for i in range(len(edges)):
        unit = graph[edges[i][0]][edges[i][1]]['object']
        pos = (unit.ul.x, unit.ul.y - size)
        text = seq.predicted_string[i][0].split('_')[0]
        draw.text(pos, text, fill='rgb(0, 0, 0)', font=font)
        draw.rectangle((unit.ul.x, unit.ul.y, unit.lr.x, unit.lr.y), outline='rgb(0, 0, 0)')

    image.save(fname)


def align_breaks_fitness(syllable_groups, group_lengths, syl_lengths):

    cur_pos = 0
    cost = 0
    for i, x in enumerate(syllable_groups):
        new_sum = sum(syl_lengths[cur_pos:cur_pos + x])
        cur_pos += x
        cost += abs(new_sum - group_lengths[i])

    return round(cost / len(syllable_groups))


def text_unit_method():

    # filenames = os.listdir('./png')
    # filenames = ['CF-011_3']
    # for fn in filenames:

    print('processing ' + filename + '...')

    raw_image = gc.load_image('./png/' + filename + '.png')
    image = preprocess_image(raw_image)
    cc_lines, lines_peak_locs = identify_text_lines(image)
    graph, all_peak_locs, all_line_images, projections = segment_lines_build_graph(image, cc_lines)
    transcript_string = parse_transcript('./png/' + filename + '.txt')

    # normalize features over all units
    prototypes = textUnit.get_prototypes()
    manuscript_units = [graph[x[0]][x[1]]['object'] for x in graph.edges]

    print('normalizing features...')
    all_units = manuscript_units + prototypes.values()
    for fk in all_units[0].features.keys():
        avg = np.mean([x.features[fk] for x in all_units])
        std = np.std([x.features[fk] for x in all_units])

        for n in range(len(all_units)):
            all_units[n].features[fk] = (all_units[n].features[fk] - avg) / std

    # rough and bad way to get very first node in the graph.
    first_node = min([x for x in graph.nodes if x[0] == 0], key=lambda x: x[1])

    sort_nodes = sorted(graph.nodes)
    sequences = [textUnit.unitSequence(seq=[sort_nodes[x]]) for x in range(0, len(sort_nodes), 1)]

    # single method that updates state of sequence
    # sequences = [textUnit.unitSequence(seq=[first_node])]

    completed_sequences = []

    # loop for evolving sequences
    for i in range(len(sort_nodes)):

        debug_str = ''

        # in each iteration, only evolve those with the smallest char index
        min_char_index = min(x.char_index for x in sequences)
        branch_sequences = [x for x in sequences if x.char_index == min_char_index]
        keep_sequences = [x for x in sequences if not x.char_index == min_char_index]

        debug_str += 'branching {} seqs, '.format(len(branch_sequences))

        # get possible branches from all sequences with min char index
        modified_sequences = []
        for j in branch_sequences:
            branches, status = get_branches_of_sequence(j, graph)
            if not (status or branches):
                continue
            if branches and (not status):
                max_num_sequences -= 1
                completed_sequences.append(j)
                continue
            modified_sequences += branches

        # add modified branches back to the list of unmodified sequences
        sequences = keep_sequences + modified_sequences

        if not sequences:
            break

        # an opimization: if two or more sequences have reached the same cut on the manuscript
        # and have the same char_index then only the one with lowest cost needs to be kept, since
        # the others could not possibly be better alignments. this works in addition to the
        # leaderboard system
        sequences.sort(key=lambda x: x.equivalent())
        filtered_sequences = []
        for k, group in iter.groupby(sequences, lambda x: x.equivalent()):
            filtered_sequences.append(min(group, key=lambda x: x.cost()))

        debug_str += 'filtered {} seqs, '.format(len(sequences) - len(filtered_sequences))

        # filter by cost: keep only the n sequences of lowest cost
        filtered_sequences.sort(key=lambda x: x.cost())
        max_seq = min(max_num_sequences, len(filtered_sequences))
        sequences = list(filtered_sequences[:max_seq - 1])

        # print(sequences[0].predicted_string)
        debug_str += '{} remain, '.format(len(sequences))
        debug_str += 'lowest cost {}.'.format(round(sequences[0].cost(), 3))
        print(debug_str)

    for i, s in enumerate(completed_sequences):
        print(i, s)
    with_lines = draw_lines(image, lines_peak_locs)
    test_text(image, completed_sequences[0], graph)


def draw_blob_alignment(alignment_groups, transcript_string,
                        cc_groups, gamera_image, size=55, fname='testimg.png'):
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


if __name__ == "__main__":
    print('processing ' + filename + '...')

    raw_image = gc.load_image('./png/' + filename + '.png')
    image = preprocess_image(raw_image)
    cc_lines, lines_peak_locs = identify_text_lines(image)

    cc_groups = []
    for x in cc_lines:
        cc_groups += group_ccs(x)

    group_lengths = []
    for g in cc_groups:
        width = sum([x.ncols for x in g])
        group_lengths.append(width)

    transcript_string, words_begin = parse_transcript('./png/' + filename + '.txt', syllables=True)
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

    # parameters for dynprogramming approach
    sequences = [[]]
    completed_sequences = []
    max_blob_sequences = 2500

    # iterating over blobs
    for i, gl in enumerate(group_lengths):
        print('sequences begin length', len(sequences))
        print(sequences[0])

        new_sequences = []
        print('group length', gl)

        branches = []
        for seq in sequences:
            # consider appending 1 to this sequence, then 2, and so on until it gets unreasonable

            # if gl <= min_letter_width:
            #     branches += [seq + [0]]
            #     continue

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
            # print(key, group)
            scores = [(align_breaks_fitness(x, group_lengths, syl_lengths), x) for x in group]
            best_group_member = min(scores, key=lambda x: x[0])
            new_sequences.append(best_group_member)

        # after previous for loop new_sequences still has a cost associated with each sequence
        # remove all but the least costly sequences
        print('len new sequences', len(new_sequences))
        new_sequences.sort(key=lambda x: x[0])

        sequences = [x[1] for x in new_sequences][:max_blob_sequences]

        print("----")

    # best_seq = min(completed_sequences, key=lambda x: x[0])[1]
    draw_blob_alignment(sequences[0], transcript_string, cc_groups, image)

    res = []
    pos = 0
    for i, x in enumerate(sequences[0]):
        end_syl_index = pos + x
        used_syllables_indices = list(range(pos, end_syl_index))
        pos = end_syl_index
        used_syllables = '-'.join([transcript_string[j] for j in used_syllables_indices])
        syl_length_sum = sum([syl_lengths[j] for j in used_syllables_indices])
        res.append((group_lengths[i], syl_length_sum, used_syllables,  x))


def brute_force_blob_alignment(group_lengths, syl_lengths):
    # brute force greedy alignment; let's see how that works first
    alignment_groups = []
    blob_index = 0
    current_dist = 0
    for tl in syl_lengths:

        if blob_index >= len(group_lengths):
            print('ran out of blobs!')
            break

        candidate_dist = current_dist + tl
        target = group_lengths[blob_index]
        current_badness = abs(current_dist - target)
        candidate_badness = abs(candidate_dist - target)

        print(blob_index, tl, target)

        if current_badness <= candidate_badness:
            blob_index += 1
            current_dist = tl
        else:
            current_dist += tl

        alignment_groups.append(blob_index)

    return alignment_groups


# options = {
#      'node_color': 'black',
#      'node_size': 10,
#      'width': 1,
#     }

# plt.clf()
# subgraph_nodes = [x for x in graph.nodes() if x[0] < 2]
# sg = graph.subgraph(subgraph_nodes)
# nx.draw_kamada_kawai(sg, **options)
# plt.savefig('testplot.png', dpi=800)

# sdf = [x.nrows for x in ccs]
# plt.clf()
# plt.hist(sdf, log=True, bins=30)
# plt.savefig('testplot.png')

# n = 2
# asdf = [x + all_line_images[n].offset_x for x in all_peak_locs[n]]
# imsv(draw_lines(image, asdf, False))
