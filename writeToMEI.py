import xml.etree.cElementTree as ET
import numpy as np
import gamera.core as gc
gc.init_gamera()
import textSeqCompare as tsc
import ocropyTest as ocp
from PIL import Image, ImageDraw, ImageFont
reload(tsc)
reload(ocp)


# a helper function
def intersect(ul1, lr1, ul2, lr2):
    dx = min(lr1[1], lr2[1]) - max(ul1[1], ul2[1])
    dy = min(lr1[0], lr2[0]) - max(ul1[0], ul2[0])
    if (dx > 0) and (dy > 0):
        return dx*dy
    else:
        return False


def generate_id():
    str = 'm-' + hex(np.random.randint(0, 16 ** 8))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 4))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 4))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 4))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 12))[2:]
    return str


fname = 'salzinnes_11'
raw_image = gc.load_image('./png/' + fname + '_text.png')
transcript = tsc.read_file('./png/' + fname + '_transcript.txt')
syls_boxes, image, lines_peak_locs = ocp.process(raw_image, transcript, wkdir_name='test')
med_line_spacing = np.median(np.diff(lines_peak_locs))

ns = {'id': '{http://www.w3.org/XML/1998/namespace}',
    'mei': '{http://www.music-encoding.org/ns/mei}'}

tree = ET.parse('salzinnes_mei_split/CF-011.mei')
ET.register_namespace('', 'http://www.music-encoding.org/ns/mei')
root = tree.getroot()

# this dict takes in any non-root element and returns its parent
parent_map = {c: p for p in tree.iter() for c in p}

zones = root.findall('.//{}zone'.format(ns['mei']))
surface = root.findall('.//{}surface'.format(ns['mei']))[0]
id_to_bbox = {}

for zone in zones:
    id = zone.attrib[ns['id'] + 'id']
    id_to_bbox[id] = zone.attrib

syllable_elements = root.findall('.//{}syllable'.format(ns['mei']))
all_bboxes = []

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

    # for collision, translate this bounding box downwards by the height of a line
    trans_lry = lry + med_line_spacing / 2
    trans_uly = uly + med_line_spacing / 2
    all_bboxes.append([ulx, uly, lrx, lry])

    # find which text syllable bounding boxes lie beneath this one
    colliding_syls = [s for s in syls_boxes
        if intersect(s[1], s[2], (ulx, trans_uly), (lrx, trans_lry)) > 0]

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

        # add corresponding zone to surface.
        new_zone = ET.SubElement(surface, '{}zone'.format(ns['mei']))
        new_id = generate_id()
        cur_syllable.set('facs', new_id)
        new_zone.set(ns['id'] + 'id', new_id)
        new_zone.set('lrx', str(lrx))
        new_zone.set('lry', str(lry))
        new_zone.set('ulx', str(ulx))
        new_zone.set('uly', str(uly))

    last_assigned_text = id_to_colliding_text

    center_x = (ulx + lrx) / 2
    center_y = (lrx + lry) / 2

    if prev_assigned_text:
        assign_lines.append([ulx, uly, prev_assigned_text[1][0], prev_assigned_text[1][1]])

    prev_text = leftmost_colliding_text


# end loop over syllables
# remove "syllables" that just held neumes and are now duplicates
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
    draw.line([al[0], al[1], al[2], al[3]], fill='black', width=10)


im.save('testimg_{}.png'.format(fname))
im.show()
