import gamera.core as gc
import matplotlib.pyplot as plt
from gamera.plugins.image_utilities import union_images

import numpy as np
gc.init_gamera()

#cut_tolerance_x should be large enough to totally discourage vertical slices
#cut_tolerance_y should be smaller than the vertical space between lines of text
#cut_tolerance_noise allows some amount of noise to be seen as a `gap` for cutting thru
input_image = gc.load_image('./CF-011_3.png')
cut_tolerance_x = 1000
cut_tolerance_y = 10
cut_tolerance_noise = 300
noise_area_thresh = 200         #ignore connected components with area smaller than this
prune_small_cuts_tolerance = 2
below_cuts_tolerance = 0.5

one_bit = input_image.to_onebit()
angle,tmp = one_bit.rotation_angle_projections()
onebit = one_bit.rotate(angle = angle, order = 3)

cuts = onebit.projection_cutting(cut_tolerance_x,cut_tolerance_y,cut_tolerance_noise,1)

#reject sufficiently small cuts likely to have captured noise;
#reject all cuts whose vertical height is sufficiently small.
#size threshold: Median Absolute Deviation (mad_height) * pruning tolerance.

med_height = np.median([x.nrows for x in cuts])
mad_height = np.median([abs(x.nrows - med_height) for x in cuts])

cuts[:] = [x for x in cuts if x.nrows > med_height - (mad_height * prune_small_cuts_tolerance)]

# #cuts contains very strict bounding boxes for each line of text that cut off
# #the top and bottom of each character; want to expand these boxes just enough
# #to totally contain each line of text, but no more.
# rectangles = []
# for n in range(0,len(cuts)):
#
#     add_to_bottom = int(cuts[n].nrows * below_cuts_tolerance)
#
#     if n == 0:
#         new_y_offset = 0
#     else:
#         new_y_offset = cuts[n-1].offset_y + cuts[n-1].nrows + add_to_bottom
#
#     point_ul = gc.Point(
#         cuts[n].offset_x,
#         new_y_offset - add_to_bottom
#         )
#     point_lr = gc.Point(
#         cuts[n].offset_x + cuts[n].ncols,
#         cuts[n].offset_y + cuts[n].nrows + add_to_bottom
#         )
#
#     rectangles.append( gc.SubImage(onebit,point_ul,point_lr))

#remove any connected components that do not extend into the cut portion

components = one_bit.cc_analysis()
components[:] = [c for c in components if c.nrows * c.ncols > noise_area_thresh]

cc_lines = []

def bases_coincide(slice_offset,slice_nrows,comp_offset,comp_nrows):
    """
    A helper function that returns true if the two intervals specified in the
    arguments overlap
    """

    component_base = comp_offset + comp_nrows
    component_height = min(slice_nrows,comp_nrows)

    range1 = range(slice_offset, slice_offset + slice_nrows)
    range2 = range(component_base - component_height, component_base)
    check = set(range1).intersection(range2)

    return bool(check)

for cut in cuts:

    cut_top = cut.offset_y
    cut_bottom = cut.offset_y + cut.nrows
    cut_mid = int(cut.offset_y + (cut.nrows * 0.5))

    # for n in range(len(components)):
    #
    #     comp_top = components[n].offset_y
    #     comp_bottom = components[n].offset_y + components[n].nrows
    #
    #     is_comp_top_inside = (comp_top > cut_top) and (comp_top < cut_bottom)
    #     is_comp_bottom_inside = (comp_bottom > cut_top) and (comp_bottom < cut_bottom)
    #     is_in_this_line = is_comp_top_inside or is_comp_bottom_inside
    #
    #     # is_in_this_line = comp_top < cut_mid and comp_bottom > cut_mid
    #
    #     #print("top: {} bottom: {} mid: {} isin: {}".format(
    #     #    comp_top,comp_bottom,cut_mid,is_in_this_line))
    #
    #     if is_in_this_line:s
    #         res.append(components[n])

    res = [x for x in components if
        bases_coincide(cut.offset_y,cut.nrows,x.offset_y,x.nrows)]

    res = sorted(res,key=lambda x: x.offset_x)
    cc_lines.append(res)

def imsv(img):
    if type(img) == list:
        union_images(img).save_image("testimg.png")
    else:
        img.save_image("testimg.png")


def plot(inp):
    plt.clf()
    asdf = plt.plot(inp,c='black',linewidth=1.0)
    plt.savefig("testplot.png",dpi=800)
