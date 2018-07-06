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
despeckle_amt = 100             #an int in [1,100]: ignore ccs with area smaller than this
noise_area_thresh = 500        #an int in : ignore ccs with area smaller than this

filter_size = 20                #size of moving-average filter used to smooth projection
prominence_tolerance = 0.85      #y-axis projection peaks must be at least this prominent

collision_strip_size = 50       #in [0,inf]; amt of each cc to consider when clipping
horizontal_gap_tolerance = 50   #value in pixels

def _bases_coincide(hline_position, comp_offset, comp_nrows, collision = collision_strip_size):
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

def _group_ccs(cc_list, gap_tolerance = horizontal_gap_tolerance):
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

def _exhaustively_bunch_ccs(cc_list, gap_tolerance = horizontal_gap_tolerance, max_num_ccs = 4):
    '''
    given a list of connected components on a single line (assumed to be in order from left
    to right), groups them into all possible bunches of up to max_num_ccs consecutive
    components. bunches with gaps larger than horizontal_gap_tolerance are not added.
    '''

    cc_copy = cc_list[:]
    result = []

    for n in range(1,max_num_ccs):
        next_group = [cc_copy[x:x + n] for x in range(len(cc_copy) - n + 1)]

        #for each member of this group, decide if it should be added; make sure the connected
        #components are separated by less than horizontal_gap_tolerance

        for g in next_group:
            cc_end = [x.offset_x + x.ncols for x in g[:-1]]
            cc_begin = [x.offset_x for x in g[1:]]
            gaps = [cc_begin[x] - cc_end[x] for x in range(n-1)]

            if not any([x >= horizontal_gap_tolerance for x in gaps]):
                result.append(g)

    #result += [[x] for x in cc_copy]
    return result

def _bounding_box(cc_list):
    '''
    given a list of connected components, finds the smallest
    bounding box that encloses all of them.
    '''

    upper = [x.offset_y for x in cc_list]
    lower = [x.offset_y + x.nrows for x in cc_list]
    left = [x.offset_x for x in cc_list]
    right = [x.offset_x  + x.ncols for x in cc_list]

    ul = gc.Point(min(left),min(upper))
    lr = gc.Point(max(right),max(lower))

    return ul, lr

def _calculate_peak_prominence(data,index):
    '''
    returns the log of the prominence of the peak at a given index in a given dataset. peak
    prominence gives high values to relatively isolated peaks and low values to peaks that are
    in the "foothills" of large peaks.
    '''
    current_peak = data[index]

    #ignore values at either end of the dataset or values that are not local maxima
    if (index == 0 or
        index == len(data) - 1 or
        data[index - 1] > current_peak or
        data[index + 1] > current_peak):
        return 0

    #by definition, the prominence of the highest value in a dataset is equal to the value itself
    if current_peak == max(data):
        return np.log(current_peak)

    #find index of nearest maxima which is higher than the current peak
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
    left_distance =  index - closest_left_ind

    if (right_distance) > (left_distance):
        closest = closest_left_ind
    else:
        closest = closest_right_ind

    #find the value at the lowest point between the nearest higher peak (the key col)
    lo = min(closest,index)
    hi = max(closest,index)
    between_slice = data[lo:hi]
    key_col = min(between_slice)

    prominence = np.log(data[index] - key_col + 1)

    return prominence

def _identify_text_lines_and_group_ccs(input_image):
    #find likely rotation angle and correct
    print('correcting rotation...')
    image_grey = input_image.to_greyscale()
    image_bin = image_grey.to_onebit()
    angle,tmp = image_bin.rotation_angle_projections()
    image_bin = image_bin.rotate(angle = angle)
    #image_bin = image_bin.erode_dilate(2,1,1)
    image_bin.despeckle(despeckle_amt)

    #compute y-axis projection of input image and filter with sliding window average
    print('smoothing projection...')
    project = image_bin.projection_rows()
    smoothed_projection = [0] * len(project)

    for n in range(filter_size, len(project) - filter_size):
        vals = project[n - filter_size : n + filter_size + 1]
        smoothed_projection[n] = np.mean(vals)

    #calculate normalized log prominence of all peaks in projection
    print('calculating log prominence of peaks...')
    prominences = [(i, _calculate_peak_prominence(smoothed_projection, i)) for i in range(len(smoothed_projection))]
    prom_max = max([x[1] for x in prominences])
    prominences[:] = [(x[0], x[1] / prom_max) for x in prominences]
    peak_locations = [x[0] for x in prominences if x[1] > prominence_tolerance]

    #perform connected component analysis and remove sufficiently small ccs and ccs that are too
    #tall; assume these to be ornamental letters
    print('connected component analysis...')
    components = image_bin.cc_analysis()
    med_comp_height = np.median([c.nrows for c in components])
    components[:] = [c for c in components if c.black_area()[0] > noise_area_thresh and c.nrows < (med_comp_height * 2)]

    #using the peak locations found earlier, find all connected components that are intersected by a
    #horizontal strip at either edge of each line. these are the lines of text in the manuscript
    print('intersecting connected components with text lines...')
    cc_lines = []
    for line_loc in peak_locations:
        res = [x for x in components if _bases_coincide(line_loc, x.offset_y,x.nrows)]
        res = sorted(res,key=lambda x: x.offset_x)
        cc_lines.append(res)

    #if a single connected component appears in more than one cc_line, give priority to the line
    #that is closer to the cemter of the component's bounding box
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

    #remove all empty lines from cc_lines in case they've been created by previous steps
    cc_lines[:] = [x for x in cc_lines if bool(x)]

    #group together connected components on the same line into bunches assumed
    #to be composed of whole words or multiple words

    print('bunching connected components....')
    manuscript_syllables = []

    #manuscript_syllables is a list of lists; each top-level entry corresponds to another line in the
    #manuscript, and each sublist contains bunches of connected components
    for n in range(len(cc_lines)):
        bunches = _exhaustively_bunch_ccs(cc_lines[n])
        syllable_list = []

        for b in bunches:
            bunch_image = union_images(b)
            syllable_list.append(syllable.Syllable(image = bunch_image))

        manuscript_syllables += syllable_list

    return {'image':image_bin,
            'ccs':manuscript_syllables,
            'peaks':peak_locations,
            'projection':smoothed_projection}

def _parse_transcript_syllables(filename):

    res = []

    file = open(filename,'r')
    lines = (file.readlines())
    file.close()

    for l in lines:
        phrase = re.split('[\s-]', l)
        res.append([syllable.Syllable(text = x.lower()) for x in phrase if x])

    return res

#LOCAL HELPER FUNCTIONS - DON'T END UP IN RODAN
def imsv(img,fname = ''):
    if type(img) == list:
        union_images(img).save_image("testimg =" + fname + ".png")
    elif type(img) == syllable.Syllable:
        img.image.save_image("testimg " + fname + ".png")
    else:
        img.save_image("testimg " + fname + ".png")

def plot(inp):
    plt.clf()
    asdf = plt.plot(inp,c='black',linewidth=0.5)
    plt.savefig("testplot.png",dpi=800)

def draw_horizontal_lines(image,line_locs):

    for l in line_locs:
        start = gc.FloatPoint(0,l)
        end = gc.FloatPoint(image.ncols,l)
        image.draw_line(start, end, 1, 5)

if __name__ == "__main__":

    #filenames = os.listdir('./png')
    filenames = ['CF-011_3']

    for fn in filenames:

        print('processing ' + fn + '...')

        image = gc.load_image('./png/' + fn + '.png')
        res = _identify_text_lines_and_group_ccs(image)
        image = res['image']
        manuscript_syllables = res['ccs']
        peak_locs = res['peaks']

        transcript_slbs = _parse_transcript_syllables('./png/' + fn + '.txt')

        # for syl in manuscript_syllables:
        #     image.draw_hollow_rect(syl.ul,syl.lr,1,5)
        # draw_horizontal_lines(image,peak_locs)
        # imsv(image,fn)

        print('performing comparisons...')
        group = transcript_slbs[0]
        pairs = []
        for i in group:
            c = [syllable.compare(g,i) for g in manuscript_syllables]
            mindex, temp = min(enumerate(c), key = lambda p: p[1][0])
            v = manuscript_syllables[mindex]
            pairs.append((i,v,temp))

        ind = 7
        imsv([pairs[ind][0].image, pairs[ind][1].image])

# plt.clf()
# plt.scatter([x[0] for x in prominences],[x[1] for x in prominences],s=5)
# plt.plot([x / max(smoothed_projection) for x in smoothed_projection],linewidth=1,color='k')
# plt.savefig("testplot " + filename + ".png",dpi=800)
