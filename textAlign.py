import gamera.core as gc
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images

import numpy as np
gc.init_gamera()


input_image = gc.load_image('./CF-014_3.png')

despeckle_amt = 100             #an int in [1,100]: ignore ccs with area smaller than this
noise_area_thresh = 200
collision_strip_size = 50         #in [0,inf]; amt of each cc to consider when clipping
horizontal_gap_tolerance = 25   #value in pixels

filter_size = 20 #to either side

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

    current_peak = smoothed_projection[ index ]

    if (index == 0 or
        index == len(smoothed_projection) - 1 or
        data[index - 1] > current_peak or
        data[index + 1] > current_peak):
        return 0

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

    lo = min(closest,index)
    hi = max(closest,index)
    between_slice = data[lo:hi]
    key_col = min(between_slice)

    prominence = np.log(data[index] - key_col + 1)

    return prominence

#find likely rotation angle and correct
image_bin = input_image.to_onebit()
angle,tmp = image_bin.rotation_angle_projections()
onebit = image_bin.rotate(angle = angle)
image_bin.despeckle(despeckle_amt)

project = image_bin.projection_rows()
smoothed_projection = [0] * len(project)

for n in range(filter_size, len(project) - filter_size):

    vals = project[n - filter_size : n + filter_size + 1]
    smoothed_projection[n] = np.mean(vals)

#one-dimensional topographic prominence
#lowest point between highest point and next-highest point

prominences = [(i,_calculate_peak_prominence(smoothed_projection,i)) for i in range(len(smoothed_projection))]
prom_max = max([x[1] for x in prominences])
peak_locations = [x[0] for x in prominences if x[1] > prom_max - 2]

components = image_bin.cc_analysis()
components[:] = [c for c in components if c.black_area > noise_area_thresh]

cc_lines = []

for line_loc in peak_locations:

    res = [x for x in components if _bases_coincide(line_loc,x.offset_y,x.nrows)]

    res = sorted(res,key=lambda x: x.offset_x)
    cc_lines.append(res)

cc_lines[:] = [x for x in cc_lines if bool(x)]

#if a single connected component appears in more than one cc_line, give priority to the line
#that is closer to the component's center of mass

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

cc_groups = [None] * len(cc_lines)
gap_sizes = [None] * len(cc_lines)

for n in range(len(cc_lines)):
    cc_groups[n], gap_sizes[n] = _group_ccs(cc_lines[n])

for group_list in cc_groups:
    for group in group_list:
        ul, lr = _bounding_box(group)
        image_bin.draw_hollow_rect(ul,lr,1,5)


#LOCAL HELPER FUNCTIONS - DON'T END UP IN RODAN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def imsv(img):
    if type(img) == list:
        union_images(img).save_image("testimg.png")
    else:
        img.save_image("testimg.png")

def plot(inp):
    plt.clf()
    asdf = plt.plot(inp,c='black',linewidth=0.5)
    plt.savefig("testplot.png",dpi=800)

def draw_horizontal_lines(image,line_locs):

    for l in line_locs:
        start = gc.FloatPoint(0,l)
        end = gc.FloatPoint(image.ncols,l)
        image.draw_line(start, end, 1, 5)

draw_horizontal_lines(image_bin,peak_locations)
imsv(image_bin)

# plt.clf()
# plt.scatter([x[0] for x in prominences],[x[1] for x in prominences])
# plt.savefig("testplot.png",dpi=800)
