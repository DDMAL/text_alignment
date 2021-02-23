from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools as iter
import os
import re
import cv2 as cv


# PARAMETERS FOR PREPROCESSING
saturation_thresh = 0.9
sat_area_thresh = 150
soften_amt = 5          # size of gaussian blur to apply before taking threshold
fill_holes = 3          # size of kernel used for morphological operations when despeckling

# PARAMETERS FOR TEXT LINE SEGMENTATION
filter_size = 30                # size of moving-average filter used to smooth projection
prominence_tolerance = 0.70     # log-projection peaks must be at least this prominent
collision_strip_scale = 1       # in [0,inf]; amt of each cc to consider when clipping
remove_capitals_scale = 10000   # removes large ccs. turned off for now

# CC GROUPING (BLOBS)
cc_group_gap_min = 20  # any gap at least this wide will be assumed to be a space between words!
max_distance_to_staff = 200


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


def preprocess_images(input_image, soften=soften_amt, fill_holes=fill_holes, correct_rotation=True):
    '''
    use gamera to do some denoising, etc on the text layer before attempting text line
    segmentation
    '''

    gray_img = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray_img, (5, 5), 0)
    ret3, img_bin = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((fill_holes, fill_holes), np.uint8)
    cl = cv.morphologyEx(img_bin, cv.MORPH_CLOSE, kernel)
    img_eroded = cv.morphologyEx(cl, cv.MORPH_OPEN, kernel)

    line_strips, lines_peak_locs, proj = identify_text_lines(img_eroded)

    angle = 0

    return img_bin, img_eroded, angle


def identify_text_lines(img, widen_strips_factor=1):
    '''
    finds text lines on preprocessed image. step-by-step:
    1. find peak locations of vertical projection
    2. draw horizontal white lines between peak locations to make totally sure that lines are
        unconnected (ornamental letters can often touch the line above them)
    3. connected component analysis
    4. break into neat rows of connected components that each intersect the same horizontal line
    '''

    # compute y-axis projection of input image and filter with sliding window average
    project = np.clip(255 - img, 0, 1).sum(1)
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
        img[idx, :] = 255

    diff_proj_peaks = find_peak_locations(np.abs(np.diff(smoothed_projection)))

    line_strips = []
    for p in peak_locations:
        # get the largest diff-peak smaller than this peak, and the smallest diff-peak that's larger
        lower_peaks = [x for x in diff_proj_peaks if x < p]
        lower_bound = max(lower_peaks) if len(lower_peaks) > 0 else 0

        higher_peaks = [x for x in diff_proj_peaks if x > p]
        higher_bound = min(higher_peaks) if len(higher_peaks) > 0 else 0

        if higher_bound and not lower_bound:
            lower_bound = p + (p - higher_bound)
        elif lower_bound and not higher_bound:
            higher_bound = p + (p - lower_bound)

        # extend bounds of strip slightly away from peak location, for safety (diacritics, etc)
        lower_bound -= int((p - lower_bound) * widen_strips_factor)
        higher_bound += int((higher_bound - p) * widen_strips_factor)

        # tighten up strip by finding bounding box around contents
        mask = np.zeros(img.shape, np.uint8)
        mask[lower_bound:higher_bound, :] = 255 - img[lower_bound:higher_bound, :]
        strip_bounds = cv.boundingRect(mask)

        line_strips.append(strip_bounds)

    return line_strips, peak_locations, smoothed_projection


def save_preproc_image(image, line_strips, lines_peak_locs, fname):
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

    fnames = ['salzinnes_378', 'salzinnes_222', 'salzinnes_315', 'salzinnes_160']

    for fname in fnames:
        print('processing {}...'.format(fname))
        raw_image = cv.imread('./png/{}_text.png'.format(fname))

        img_bin, img_eroded, angle = preprocess_images(raw_image, soften=soften_amt, fill_holes=3)

        # cv.imwrite('test.png', img_eroded)

        line_strips, lines_peak_locs, proj = identify_text_lines(img_eroded)
        save_preproc_image(img_bin, line_strips, lines_peak_locs, fname)

    # plt.clf()
    # plt.plot(proj)
    # for x in lines_peak_locs:
    #     plt.axvline(x=x, linestyle=':')
    # plt.show()
    #
    # diff_proj_peaks = find_peak_locations(np.abs(np.diff(proj)))
    # plt.clf()
    # plt.plot(np.diff(proj))
    # for x in diff_proj_peaks:
    #     plt.axvline(x=x, linestyle=':')
    # plt.show()
