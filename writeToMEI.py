import xml.etree.cElementTree as ET
import numpy as np
import gamera.core as gc
gc.init_gamera()
import textSeqCompare as tsc
import ocropyTest as ocp
from PIL import Image, ImageDraw, ImageFont
reload(tsc)
reload(ocp)


# layer = ET.Element('layer')
# syllable = ET.SubElement(layer, 'syllable')
# ET.SubElement(syllable, "syl").text = "Splen"
# syllable2 = ET.SubElement(layer, 'syllable')
# ET.SubElement(syllable2, "syl").text = "dor"
#
# tree = ET.ElementTree(layer)
# ET.dump(tree)

fname = 'salzinnes_11'
raw_image = gc.load_image('./png/' + fname + '_text.png')
transcript = tsc.read_file('./png/' + fname + '_transcript.txt')
syls_boxes, image, lines_peak_locs = ocp.process(raw_image, transcript, wkdir_name='test')
med_line_spacing = np.median(np.diff(lines_peak_locs))

ns = {'id': '{http://www.w3.org/XML/1998/namespace}',
    'mei': '{http://www.music-encoding.org/ns/mei}'}

tree = ET.parse('salzinnes_mei_split/CF-011.mei')
root = tree.getroot()

# this dict takes in any non-root element and returns its parent
parent_map = {c:p for p in tree.iter() for c in p}

zones = root.findall('.//{}zone'.format(ns['mei']))
id_to_bbox = {}

# get all elements that have an id
all_elements = root.findall('.//nc')
# all_elements = [el for el in all_elements if 'zone' not in el.tag]
# id_to_element = {el.attrib[ns['id'] + 'id']: el for el in all_elements}


for zone in zones:
    id = zone.attrib[ns['id'] + 'id']
    id_to_bbox[id] = zone.attrib

syllable_elements = root.findall('.//{}syllable'.format(ns['mei']))
all_bboxes = []


# a helper function for the next part:
def intersect(ul1, lr1, ul2, lr2):
    # is the top of 1 below the bottom of 2 (or vice versa)?
    tmp1 = (ul1[1] > lr2[1]) or (ul2[1] > lr1[1])
    # is the left side of 1 to the right of 2 (or vice versa)?
    tmp2 = (ul1[0] > lr2[0]) or (ul2[0] > lr1[0])
    # iff either of these are true, then there is no intersection
    return not (tmp1 or tmp2)


id_to_colliding_text = {}
cur_syllable = None
prev_text = None
prev_assigned_text = None
elements_to_remove = []
assign_lines = []
for i, se in enumerate(syllable_elements):
    # get the neume associated with this syllable
    neume = se[0]
    syl_id = se.attrib[ns['id'] + 'id']

    if not cur_syllable:
        cur_syllable = se

    assert 'neume' in neume.tag

    neume_components = neume.findall(ns['mei'] + 'nc')
    bboxes = [id_to_bbox[nc.attrib['facs']] for nc in neume_components]

    # get bounding box that surrounds every neume component in this neume
    lrx = max(int(bb['lrx']) for bb in bboxes)
    lry = max(int(bb['lry']) for bb in bboxes)
    ulx = min(int(bb['ulx']) for bb in bboxes)
    uly = min(int(bb['uly']) for bb in bboxes)

    all_bboxes.append([ulx, uly, lrx, lry])

    # for collision, extend this bounding box downwards by the height of a line
    lry += med_line_spacing / 2

    # find which text syllable bounding boxes lie beneath this one
    colliding_syls = [s for s in syls_boxes if intersect(s[1], s[2], (ulx, uly), (lrx, lry))]

    if colliding_syls:
        leftmost_colliding_text = min(colliding_syls, key=lambda x: x[1][0])
        prev_assigned_text = leftmost_colliding_text
    else:
        leftmost_colliding_text = None
    id_to_colliding_text[syl_id] = leftmost_colliding_text

    # if there is no text OR if the found text is the same as last time then the neume being
    # considered here is linked to the previous syllable.
    if (not leftmost_colliding_text) or (leftmost_colliding_text == prev_text):
        cur_syllable.append(neume)
        elements_to_remove.append(se)
    else:
        cur_syllable = se
        cur_syllable.text = leftmost_colliding_text[0]
        cur_syllable.append(neume)

    last_assigned_text = id_to_colliding_text

    center_x = (ulx + lrx) / 2
    center_y = (lrx + lry) / 2

    if prev_assigned_text:
        assign_lines.append([ulx, uly, prev_assigned_text[1][0], prev_assigned_text[1][1]])

    prev_text = leftmost_colliding_text


for el in elements_to_remove:
    parent_map[el].remove(el)

tree.write('testxml_{}.xml'.format(fname))

#############################
# -- DRAW RESULTS ON PAGE --
#############################

im = raw_image.to_greyscale().to_pil()
text_size = 70
fnt = ImageFont.truetype('Arial.ttf', text_size)
draw = ImageDraw.Draw(im)

for i, char in enumerate(syls_boxes):
    if char[0] in '. ':
        continue

    ul = char[1]
    lr = char[2]
    draw.text((ul[0], ul[1] - text_size), char[0], font=fnt, fill='gray')
    draw.rectangle([ul, lr], outline='black')
    draw.line([ul[0], ul[1], ul[0], lr[1]], fill='black', width=10)

# for i, peak_loc in enumerate(lines_peak_locs):
#     draw.text((1, peak_loc - text_size), 'line {}'.format(i), font=fnt, fill='gray')
#     draw.line([0, peak_loc, im.width, peak_loc], fill='gray', width=3)

for box in all_bboxes:
    draw.rectangle(box, outline='black')

for al in assign_lines:
    draw.line([al[0], al[1], al[2], al[3]], fill='black', width=15)


im.save('testimg_{}.png'.format(fname))
im.show()
