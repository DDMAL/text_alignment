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
    # is the top of 1 below the bottom of 2 (or vice versa)
    tmp1 = (ul1[1] > lr2[1]) or (ul2[1] > lr1[1])
    # is the left side of 1 to the right of 2 (or vice versa)
    tmp2 = (ul1[0] > lr2[0]) or (ul2[0] > lr1[0])
    # if either of these are true, then there is no intersection
    return not (tmp1 or tmp2)


for se in syllable_elements:
    # get the neume associated with this syllable
    neume = se[0]

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
    lry += med_line_spacing

    # find which text syllable bounding box lies beneath this one. if none does, then
    # this neume is assigned to the previous text syllable.
    colliding_syls = [s for s in syls_boxes if intersect(s[1], s[2], (ulx, uly), (lrx, lry))]
    print(len(colliding_syls))


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

for i, peak_loc in enumerate(lines_peak_locs):
    draw.text((1, peak_loc - text_size), 'line {}'.format(i), font=fnt, fill='gray')
    draw.line([0, peak_loc, im.width, peak_loc], fill='gray', width=3)

for box in all_bboxes:
    draw.rectangle(box, outline='black')

im.save('testimg_{}.png'.format(fname))
im.show()
