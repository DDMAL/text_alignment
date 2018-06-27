import gamera.core as gc
import matplotlib.pyplot as plt
import numpy as np
gc.init_gamera()

#cut_tolerance_x should be large enough to totally discourage vertical slices
#cut_tolerance_y should be smaller than the vertical space between lines of text
#cut_tolerance_noise allows some amount of noise to be seen as a `gap` for cutting thru
input_image = gc.load_image('./CF-012_3.png')
cut_tolerance_x = 1000
cut_tolerance_y = 10
cut_tolerance_noise = 300
noise_area_thresh = 300         #ignore connected components with area smaller than this
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

#cuts contains very strict bounding boxes for each line of text that cut off
#the top and bottom of each character; want to expand these boxes just enough
#to totally contain each line of text, but no more.
rectangles = []
for n in range(0,len(cuts)):

    add_to_bottom = int(cuts[n].nrows * below_cuts_tolerance)

    if n == 0:
        new_y_offset = 0
    else:
        new_y_offset = cuts[n-1].offset_y + cuts[n-1].nrows + add_to_bottom

    point_ul = gc.Point(
        cuts[n].offset_x,
        new_y_offset - add_to_bottom
        )
    point_lr = gc.Point(
        cuts[n].offset_x + cuts[n].ncols,
        cuts[n].offset_y + cuts[n].nrows + add_to_bottom
        )

    rectangles.append( gc.SubImage(onebit,point_ul,point_lr))

#remove any connected components that do not extend into the cut portion

components = one_bit.cc_analysis()

cc_lines = []


for cut in cuts:
    res = []

    cut_top = cut.offset_y
    cut_bottom = cut.offset_y + cut.nrows

    for comp in components:

        if (comp.nrows * comp.ncols) < noise_area_thresh:
            continue

        comp_top = comp.offset_y
        comp_bottom = comp.offset_y + comp.nrows

        is_comp_top_inside = (comp_top > cut_top) and (comp_top < cut_bottom)
        is_comp_bottom_inside = (comp_bottom > cut_top) and (comp_bottom < cut_bottom)

        is_in_this_line = is_comp_top_inside or is_comp_bottom_inside

        if is_in_this_line:
            res.append(comp)

    cc_lines.append(res)



#now, split rectangles into "words;" at least, spaces large enough to
#definitely be words

def imsv(img):
    img.save_image("testimg.png")


def plot(inp):
    plt.clf()
    asdf = plt.plot(inp,c='black',linewidth=1.0)
    plt.savefig("testplot.png",dpi=800)
