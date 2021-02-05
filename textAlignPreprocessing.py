from os.path import isfile, join
import numpy as np
# import gamera.core as gc
# gc.init_gamera()
import matplotlib.pyplot as plt
# from gamera.plugins.image_utilities import union_images
import pickle
import itertools as iter
import os
import re

# PARAMETERS FOR PREPROCESSING
saturation_thresh = 0.9
sat_area_thresh = 150
despeckle_amt = 100            # an int in [1,100]: ignore ccs with area smaller than this
noise_area_thresh = 100        # an int in : ignore ccs with area smaller than this

# PARAMETERS FOR TEXT LINE SEGMENTATION
filter_size = 30                # size of moving-average filter used to smooth projection
prominence_tolerance = 0.70     # log-projection peaks must be at least this prominent
collision_strip_scale = 1       # in [0,inf]; amt of each cc to consider when clipping
remove_capitals_scale = 10000   # removes large ccs. turned off for now

# CC GROUPING (BLOBS)
cc_group_gap_min = 20  # any gap at least this wide will be assumed to be a space between words!
max_distance_to_staff = 200


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


def identify_text_lines_old(image_bin, image_eroded):
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


def identify_text_lines(image_eroded):
    '''
    finds text lines on preprocessed image. step-by-step:
    1. find peak locations of vertical projection
    2. draw horizontal white lines between peak locations to make totally sure that lines are
        unconnected (ornamental letters can often touch the line above them)
    3. connected component analysis
    4. break into neat rows of connected components that each intersect the same horizontal line
    '''

    # compute y-axis projection of input image and filter with sliding window average
    project = np.clip(255 - image_eroded, 0, 1).sum(1)
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
        image_eroded[idx, :] = 255

    # perform connected component analysis
    num_labels, labels = cv.connectedComponents(255 - image_eroded)

    # c = Counter(labels.reshape(-1))
    # for k in c.keys():
    #     if c[k] < noise_area_thresh:
    #         labels[labels == k] = 0

    line_strips = []
    cc_lines = []
    for line_loc in peak_locations:

        # get all components that intersect this horizontal projection peak location
        int_components = list(set(labels[line_loc, :]))
        int_components.remove(0)

        int_labels = np.isin(labels, int_components).astype('uint8')
        strip_bounds = cv.boundingRect(int_labels)
        line_strips.append(strip_bounds)

    return line_strips, peak_locations, smoothed_projection


def save_preproc_image(image, line_strips, lines_peak_locs, fname):
    # color discovered CCS with unique colors
    # ccs = [j for i in cc_lines for j in i]
    # image = image.color_ccs(True)
    im = Image.fromarray(image)

    text_size = 70
    fnt = ImageFont.truetype('FreeMono.ttf', text_size)
    draw = ImageDraw.Draw(im)

    # draw lines at identified peak locations
    for i, peak_loc in enumerate(lines_peak_locs):
        draw.text((1, peak_loc - text_size), 'line {}'.format(i), font=fnt, fill='gray')
        draw.line([0, peak_loc, im.width, peak_loc], fill='gray', width=3)

    # draw rectangles around identified text lines
    for line in line_strips:
        ul = (line[0], line[1])
        lr = (line[0] + line[2], line[1] + line[3])
        draw.rectangle([ul, lr], outline='black')

    # im.show()
    im.save('test_preproc_{}.png'.format(fname))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result




if __name__ == '__main__':
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import pyplot as plt
    import numpy as np
    import cv2 as cv

    fnames = ['salzinnes_378']

    for fname in fnames:
        print(f'processing {fname}...')
        raw_image = cv.imread(f'./png/{fname}_text.png')

        gray_img = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray_img, (5, 5), 0)
        ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        cl = cv.morphologyEx(th3, cv.MORPH_CLOSE, kernel)
        eroded = cv.morphologyEx(cl, cv.MORPH_OPEN, kernel)

        # cv.imwrite('test.png', eroded)

        line_strips, lines_peak_locs, proj = identify_text_lines(eroded)



        #

        # save_preproc_image(image, line_strips, lines_peak_locs, fname)

    # plt.clf()
    # plt.plot(proj)
    # for x in lines_peak_locs:
    #     plt.axvline(x=x, linestyle=':')
    # plt.show()
