import gamera.core as gc
import matplotlib.pyplot as plt
import numpy as np
gc.init_gamera()

#cut_tolerance_x should be larger than the maximum space between words on the
#same line
#cut_tolerance_y should be smaller than the vertical space between lines of text
input_image = gc.load_image('./CF-010_3.png')
cut_tolerance_x = 100
cut_tolerance_y = 100
cut_tolerance_noise = 300
prune_small_cuts_tolerance = 5
below_cuts_tolerance = 0.5

onebit = input_image.to_onebit()
angle,tmp = onebit.rotation_angle_projections()
onebit = onebit.rotate(angle = angle, order = 3)

cuts = onebit.projection_cutting(cut_tolerance_x,cut_tolerance_y,cut_tolerance_noise,1)

#reject sufficiently small cuts likely to have captured noise;
#reject all cuts whose vertical height is sufficiently small.
#size threshold: median of all absolute distances from median.

med_height = np.median([x.nrows for x in cuts])
std_height = np.median([abs(x.nrows - med_height) for x in cuts])

cuts[:] = [x for x in cuts if x.nrows > med_height - (std_height * prune_small_cuts_tolerance)]

#the cuts are rather harsh
rectangles = []
for n in range(1,len(cuts)):

    add_to_bottom = int(cuts[n].nrows * below_cuts_tolerance)

    point_ul = gc.Point(
        cuts[n].offset_x,
        cuts[n-1].offset_y + cuts[n-1].nrows
        )
    point_lr = gc.Point(
        cuts[n].offset_x + cuts[n].ncols,
        cuts[n].offset_y + cuts[n].nrows + add_to_bottom
        )

    rectangles.append( gc.SubImage(onebit,point_ul,point_lr))

#go back and insert first row
add_to_bottom = int(cuts[0].nrows * below_cuts_tolerance)

point_ul = gc.Point(
    cuts[0].offset_x,
    cuts[0].offset_y - rectangles[0].nrows
    )
point_lr = gc.Point(
    cuts[0].offset_x + cuts[0].ncols,
    cuts[0].offset_y + cuts[0].nrows + add_to_bottom
    )

rectangles.insert(0,gc.SubImage(onebit,point_ul,point_lr))

def imsv(img):
    img.save_image("testimg.png")

def plot(inp):
    plt.clf()
    asdf = plt.plot(inp,c='black',linewidth=1.0)
    plt.savefig("testplot.png",dpi=800)
