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

def bases_coincide(slice_offset,slice_nrows,comp_offset,comp_nrows):
    """
    A helper function that takes in the vertical width of a horizontal strip
    and the vertical measurements of a connected component, and returns a value
    of True if the bottom of the connected component lies within the strip.

    If the connected component is shorter than the height of the strip, then
    we just check if any part of it lies within the strip at all.
    """

    component_base = comp_offset + comp_nrows
    component_height = min(slice_nrows,comp_nrows)

    range1 = range(slice_offset, slice_offset + slice_nrows)
    range2 = range(component_base - component_height, component_base)
    check = set(range1).intersection(range2)

    return bool(check)

#find likely rotation angle and correct
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

components = one_bit.cc_analysis()
components[:] = [c for c in components if c.nrows * c.ncols > noise_area_thresh]

cc_lines = []

for cut in cuts:

    res = [x for x in components if
        bases_coincide(cut.offset_y,cut.nrows,x.offset_y,x.nrows)]

    res = sorted(res,key=lambda x: x.offset_x)
    cc_lines.append(res)



#LOCAL HELPER FUNCTIONS - DON'T END UP IN RODAN
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def imsv(img):
    if type(img) == list:
        union_images(img).save_image("testimg.png")
    else:
        img.save_image("testimg.png")


def plot(inp):
    plt.clf()
    asdf = plt.plot(inp,c='black',linewidth=1.0)
    plt.savefig("testplot.png",dpi=800)
