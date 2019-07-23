from os.path import isfile, join
import numpy as np
import gamera.core as gc
gc.init_gamera()
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images
from gamera import graph_util
from gamera import graph
import itertools as iter
import os
import re

# PARAMETERS FOR PREPROCESSING
saturation_thresh = 0.9
sat_area_thresh = 150
despeckle_amt = 100            # an int in [1,100]: ignore ccs with area smaller than this
noise_area_thresh = 200        # an int in : ignore ccs with area smaller than this

# PARAMETERS FOR TEXT LINE SEGMENTATION
filter_size = 30                # size of moving-average filter used to smooth projection
prominence_tolerance = 0.70     # log-projection peaks must be at least this prominent
collision_strip_scale = 1       # in [0,inf]; amt of each cc to consider when clipping
remove_capitals_scale = 10000   # removes large ccs. turned off for now

# CC GROUPING (BLOBS)
cc_group_gap_min = 20  # any gap at least this wide will be assumed to be a space between words!
max_distance_to_staff = 200

# i need a custom set of colors when working on cc analysis because i'm too colorblind for the default :(
colors = [ gc.RGBPixel(150, 0, 0),
           gc.RGBPixel(0, 100, 0),
           gc.RGBPixel(0, 0, 255),
           gc.RGBPixel(250, 0, 255),
           gc.RGBPixel(50, 150, 50),
           gc.RGBPixel(0, 190, 230),
           gc.RGBPixel(230, 100, 20) ]


def vertically_coincide(hline_position, comp_offset, comp_nrows, collision, collision_scale=collision_strip_scale):
    """
    A helper function that takes in the vertical width of a horizontal strip
    and the vertical measurements of a connected component, and returns a value
    of True if any part of it lies within the strip.
    """

    collision *= collision_strip_scale

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


def find_peak_locations(data, tol=prominence_tolerance, ranked=False):
    '''
    given a vertical projection in @data, finds prominent peaks and returns their indices
    '''

    prominences = [(i, calculate_peak_prominence(data, i)) for i in range(len(data))]

    # normalize to interval [0,1]
    prom_max = max([x[1] for x in prominences])
    if prom_max == 0 or len(prominences) == 0:
        # failure to find any peaks; probably monotonically increasing / decreasing
        return []

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

    if ranked:
        peak_locs.sort(key=lambda x: x[1] * -1)
    else:
        peak_locs[:] = [x[0] for x in peak_locs]

    return peak_locs


def moving_avg_filter(data, filter_size=filter_size):
    '''
    returns a list containing the data in @data filtered through a moving-average filter of size
    @filter_size to either side; that is, filter_size = 1 gives a size of 3, filter size = 2 gives
    a size of 5, and so on.
    '''
    smoothed = np.zeros(len(data))
    for n in range(filter_size, len(data) - filter_size):
        vals = data[n - filter_size: n + filter_size + 1]
        smoothed[n] = np.mean(vals)
    return smoothed


def preprocess_images(input_image, despeckle_amt=despeckle_amt, filter_runs=1, filter_runs_amt=2, correct_rotation=True):
    '''
    use gamera to do some denoising, etc on the text layer before attempting text line
    segmentation
    '''

    image_bin = input_image.to_onebit()

    image_bin.despeckle(despeckle_amt)
    image_bin.invert()
    image_bin.despeckle(despeckle_amt)
    image_bin.invert()

    # keep only colored ccs above a certain size
    ccs = image_bin.cc_analysis()
    for c in ccs:
        area = c.nrows
        if sat_area_thresh < area:
            c.fill_white()

    # image_bin = input_image.to_onebit().subtract_images(image_bin)

    # find likely rotation angle and correct
    angle, tmp = image_bin.rotation_angle_projections(-6, 6)
    if correct_rotation:
        image_bin = image_bin.rotate(angle=angle)

    image_bin.reset_onebit_image()

    image_eroded = image_bin.image_copy()

    for i in range(filter_runs):
        image_eroded.filter_short_runs(filter_runs_amt, 'black')
        image_eroded.filter_narrow_runs(filter_runs_amt, 'black')

    return image_bin, image_eroded, angle


def identify_text_lines(image_bin, image_eroded):
    '''
    finds text lines on preprocessed image. step-by-step:
    1. find peak locations of vertical projection
    2. draw horizontal white lines between peak locations to make totally sure that lines are
        unconnected (ornamental letters can often touch the line above them)
    3. connected component analysis
    4. break into neat rows of connected components that each intersect the same horizontal line
    5. deal with some pathological cases (empty lines, doubled lines, etc)
    '''

    # compute y-axis projection of input image and filter with sliding window average
    print('finding projection peaks...')
    project = image_eroded.projection_rows()
    smoothed_projection = moving_avg_filter(project, filter_size)

    # calculate normalized log prominence of all peaks in projection
    peak_locations = find_peak_locations(smoothed_projection)

    # draw a horizontal white line at the local minima of the vertical projection. this ensures
    # that every connected component can intersect at most one text line.
    for i in range(len(peak_locations) - 1):
        start = peak_locations[i]
        end = peak_locations[i + 1]
        idx = np.argmin(smoothed_projection[start:end])
        idx += start
        image_eroded.draw_line((0, idx), (image_eroded.ncols, idx), 0, 2)

    # perform connected component analysis and remove sufficiently small ccs and ccs that are too
    # tall; assume these to be ornamental letters
    print('connected component analysis...')
    components = image_eroded.cc_analysis()

    for c in components:
        if c.black_area()[0] < noise_area_thresh:
            c.fill_white()

    components[:] = [c for c in components if c.black_area()[0] > noise_area_thresh]

    med_comp_height = np.median([c.nrows for c in components])

    components[:] = [c for c in components if c.nrows < (med_comp_height * remove_capitals_scale)]

    # using the peak locations found earlier, find all connected components that are intersected by
    # a horizontal strip at either edge of each line. these are the lines of text in the manuscript

    line_strips = []

    cc_median_height = np.median([x.nrows for x in components])
    cc_lines = []
    for line_loc in peak_locations:
        res = [x for x in components if vertically_coincide(line_loc, x.offset_y, x.nrows, cc_median_height)]

        ulx = min(s.ul.x for s in res)
        uly = min(s.ul.y for s in res)
        lrx = max(s.lr.x for s in res)
        lry = max(s.lr.y for s in res)

        strip = image_bin.subimage((ulx, uly), (lrx, lry))
        line_strips.append(strip)

    # if a single connected component appears in more than one cc_line, give priority to the line
    # that is closer to the center of the component's bounding box
    # for n in range(len(cc_lines) - 1):
    #     intersect = set(cc_lines[n]) & set(cc_lines[n + 1])
    #
    #     # if most of the ccs are shared between these lines, just delete one of them
    #     if len(intersect) > (0.5 * min(len(cc_lines[n]), len(cc_lines[n + 1]))):
    #         cc_lines[n] = []
    #         continue
    #
    #     for i in intersect:
    #
    #         box_center = i.offset_y + (i.nrows / 2)
    #         distance_up = abs(peak_locations[n] - box_center)
    #         distance_down = abs(peak_locations[n + 1] - box_center)
    #
    #         if distance_up > distance_down:
    #             cc_lines[n].remove(i)
    #             # print('removing up')
    #         else:
    #             cc_lines[n+1].remove(i)
    #             # print('removing down')

    # remove all empty lines from cc_lines in case they've been created by previous steps
    # cc_lines[:] = [x for x in cc_lines if bool(x)]

    return line_strips, peak_locations, smoothed_projection


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

    return result, gap_sizes


def smear_lines(image, points_factor=0.005):
    filt = image.image_copy()
    print('connected component analysis...')
    components = filt.cc_analysis()

    for c in components:
        if c.black_area()[0] < noise_area_thresh:
            c.fill_white()
    med_comp_width = int(np.median([c.ncols for c in components
        if c.black_area()[0] > noise_area_thresh]))
    med_comp_height = int(np.median([c.nrows for c in components
        if c.black_area()[0] > noise_area_thresh]))

    filt.reset_onebit_image()

    filt.filter_narrow_runs(med_comp_width * 2, 'white')
    filt.filter_narrow_runs(med_comp_width * 2, 'black')
    # filt.filter_narrow_runs(med_comp_width * 2, 'white')
    # filt.filter_narrow_runs(med_comp_width * 2, 'black')

    # gamera crashes when calculating a convex hull on very skinny ccs, so as a workaround
    # just get rid of them altogether at the outset
    # filt.filter_short_runs(med_comp_height // 8, 'black')
    # filt.filter_short_runs(med_comp_height // 8, 'white')

    smear = filt.cc_analysis()

    # remove smeared rows that are too short
    med_smear_height = np.median([c.nrows for c in smear])
    smear = [s for s in smear if s.nrows >= med_smear_height / 2]
    overlap_thresh = int(med_smear_height / 2)

    g = graph.Graph(graph.FLAG_DAG)
    for a, b in iter.product(smear, smear):
        # ensure that a is to the left of b on the page
        if a.lr_x > b.ul_x:
            continue
        # ensure that their midpoints are less than a line width apart
        # if abs((a.lr_y + a.ul_y) - (b.lr_y + b.ul_y)) > 2 * med_smear_height:
        #     continue
        # ensure that a horizontally overlaps b
        overlap_amt = min(a.lr_y - b.ul_y, b.ll_y - a.ur_y)
        # print(overlap_amt)
        if overlap_amt < overlap_thresh:
            continue
        left_mid = np.array([a.lr_x + a.ul_x, a.lr_y + a.ur_y]) / 2
        right_mid = np.array([b.lr_x + b.ul_x, b.ll_y + b.ul_y]) / 2
        val = np.linalg.norm(left_mid - right_mid)
        g.add_edge(a, b, val, True)
        # filt.draw_line(left_mid, right_mid, 1, 10)

    # want to get to a point where every node has either 1 or 0 edges set to True
    for n in g.get_nodes():

        in_edges = [x for x in g.get_edges() if x.to_node == n]
        trues = [x.label for x in in_edges]
        if sum(trues) > 1:
            for x in in_edges:
                x.label = False
            choose = min(in_edges, key=lambda x: x.cost)
            choose.label = True

        out_edges = [x for x in n.edges]
        trues = [x.label for x in out_edges]
        if sum(trues) > 1:
            for x in out_edges:
                x.label = False
            choose = min(out_edges, key=lambda x: x.cost)
            choose.label = True

    for e in g.get_edges():
        if not e.label:
            continue
        a = e.from_node.data
        b = e.to_node.data
        left_mid = np.array([a.lr_x + a.ul_x, a.lr_y + a.ur_y]) / 2
        right_mid = np.array([b.lr_x + b.ul_x, b.ll_y + b.ul_y]) / 2
        filt.draw_line(left_mid, right_mid, 1, 10)

    filt.reset_onebit_image()
    hulls = filt.image_copy()
    hulls.fill_white()
    smear = filt.cc_analysis()
    # for s in smear:
    #     points = s.convex_hull_as_points()
    #     for i in range(len(points)):
    #         hulls.draw_line(points[i-1] + s.ul, points[i] + s.ul, 1, 1)

    filt_color = filt.graph_color_ccs(smear, colors, 1)

    imp = image.to_rgb().to_numpy().astype('float')
    flp = filt_color.to_rgb().to_numpy().astype('float')
    add = (imp + flp) / 2
    comb = Image.fromarray(add.astype('uint8'))
    # comb.show()
    return comb


def strip_projections(image, num_strips=30):

    newimg = image.image_copy()

    bottom_contour = image.contour_bottom()
    left_side = min([i for i in range(len(bottom_contour) - 1)
        if bottom_contour[i + 1] != np.inf
        and bottom_contour[i] == np.inf])
    right_side = max([i for i in range(len(bottom_contour) - 1)
        if bottom_contour[i + 1] == np.inf
        and bottom_contour[i] != np.inf])

    vert_strips = []
    strip_dim = gc.Dim(1 + (right_side - left_side) // num_strips, image.height)
    for i in range(num_strips):
        ul = gc.Point(left_side + strip_dim.ncols * i, 0)
        subimg = image.subimage(ul, strip_dim)
        vert_strips.append(subimg)

    strip_centers = [int((x + 0.5) * strip_dim.ncols + left_side) for x in range(num_strips)]
    strip_projections = [x.projection_rows() for x in vert_strips]
    strip_proj_smooth = [moving_avg_filter(x) for x in strip_projections]
    strip_log = [np.log(x / (max(x) + 1) + 1) for x in strip_proj_smooth]
    anchor_points = []
    # find runs of consecutive zeroes and take points in middle of these runs: anchors
    for s in strip_log:
        iszero = np.concatenate(([0], np.less(s, 0.2).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        anchors = [int((x[0] + x[1]) / 2) for x in ranges]
        anchor_points.append(anchors)

    for i, x in enumerate(strip_centers):
        for y in anchor_points[i]:
            newimg.draw_marker(gc.Point(x, y), 10, 3, 1)

    newimg.to_rgb().to_pil().save('aaaaaaa.png')


def save_preproc_image(image, cc_strips, lines_peak_locs, fname):
    # color discovered CCS with unique colors
    # ccs = [j for i in cc_lines for j in i]
    # image = image.color_ccs(True)
    im = image.to_rgb().to_pil()

    text_size = 70
    fnt = ImageFont.truetype('FreeMono.ttf', text_size)
    draw = ImageDraw.Draw(im)

    # draw lines at identified peak locations
    for i, peak_loc in enumerate(lines_peak_locs):
        draw.text((1, peak_loc - text_size), 'line {}'.format(i), font=fnt, fill='gray')
        draw.line([0, peak_loc, im.width, peak_loc], fill='gray', width=3)

    # draw rectangles around identified text lines
    for line in cc_strips:
        unioned = line
        ul = (unioned.ul.x, unioned.ul.y)
        lr = (unioned.lr.x, unioned.lr.y)
        draw.rectangle([ul, lr], outline='black')

    # im.show()
    im.save('test_preproc_{}.png'.format(fname))


if __name__ == '__main__':
    from PIL import Image, ImageDraw, ImageFont

    fnames = ['einsiedeln_003v']

    for fname in fnames:
        print('processing {}...'.format(fname))
        raw_image = gc.load_image('./png/' + fname + '_text.png')
        image, eroded, angle = preprocess_images(raw_image)
        line_strips, lines_peak_locs, proj = identify_text_lines(image, eroded)

        save_preproc_image(image, line_strips, lines_peak_locs, fname)

    # plt.clf()
    # plt.plot(proj)
    # for x in lines_peak_locs:
    #     plt.axvline(x=x, linestyle=':')
    # plt.show()
