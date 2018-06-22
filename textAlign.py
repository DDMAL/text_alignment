import gamera.core as gc
import matplotlib.pyplot as plt
import numpy as np
gc.init_gamera()

input_image = gc.load_image('./CF-010_3.png')

#first order of business: fix rotation

#cut_tolerance_x should be larger than the maximum space between words on the
#same line
#cut_tolerance_y should be smaller than the vertical space between lines of text
cut_tolerance_x = 100
cut_tolerance_y = 100
cut_tolerance_noise = 300
prune_small_cuts_tolerance = 5

onebit = input_image.to_onebit()
angle,tmp = onebit.rotation_angle_projections()
onebit = onebit.rotate(angle = angle, order = 2)

cuts = onebit.projection_cutting(cut_tolerance_x,cut_tolerance_y,cut_tolerance_noise,1)

#reject sufficiently small cuts likely to have captured noise;
#reject all cuts whose vertical height is one standard deviation smaller than
#the mean heignt

med_height = np.median([x.nrows for x in cuts])
std_height = np.median([abs(x.nrows - med_height) for x in cuts])

cuts[:] = [x for x in cuts if x.nrows > med_height - (std_height * prune_small_cuts_tolerance)]

#find minimum projection value x such that all points in middle are above
#x and all points to either side are below.

# def projection_crossing_pts(proj,test_val):
#
#     crossing_points = []
#     for n in range(1,len(proj)):
#
#         cross_up = proj[n] - test_val
#         cross_down = proj[n-1] - test_val
#
#         if not np.sign(cross_up) == np.sign(cross_down):
#             crossing_points.append(n)
#
#     return crossing_points
#
# val_increment = 5
# test_val = 5
# crossing_point_groups = []
# val_found = False
#
# while (not val_found):



#onebit.save_image("testimg.png")

def imsv(img):
    img.save_image("testimg.png")

def plot(inp):
    plt.clf()
    asdf = plt.plot(inp,c='black',linewidth=1.0)
    plt.savefig("testplot.png",dpi=800)
