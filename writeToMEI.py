import xml.etree.cElementTree as ET
import numpy as np
import gamera.core as gc
gc.init_gamera()
import textSeqCompare as tsc
import ocropyTest as ocp
from PIL import Image, ImageDraw, ImageFont
reload(tsc)
reload(ocp)


# returns the area of the intersection between two rectangles given their upper left and
# lower right corners, or a False value if they do not intersect
def intersect(ul1, lr1, ul2, lr2):
    dx = min(lr1[1], lr2[1]) - max(ul1[1], ul2[1])
    dy = min(lr1[0], lr2[0]) - max(ul1[0], ul2[0])
    if (dx > 0) and (dy > 0):
        return dx*dy
    else:
        return False

# generates a unique ID for XML elements
def generate_id():
    str = 'm-' + hex(np.random.randint(0, 16 ** 8))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 4))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 4))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 4))[2:]
    str += '-' + hex(np.random.randint(0, 16 ** 12))[2:]
    return str

def repair_xml(input):
    pt = raw_xml.index('meiversion')
    insert = 'xmlns:xlink="http://www.w3.org/1999/xlink" '
    repaired_xml = raw_xml[:pt] + insert + raw_xml[pt:]
    return repaired_xml


fname = 'salzinnes_11'
xml_fname = 'salzinnes_mei_split/CF-017.mei'

# load data: image, transcript, MEI file
raw_image = gc.load_image('./png/' + fname + '_text.png')
transcript = tsc.read_file('./png/' + fname + '_transcript.txt')
with open(xml_fname, 'r') as f:
    raw_xml = f.read()

# forgive me for this but the xml output by pitchfinding has a namespace issue and this is the
# only way i can think of to correctly parse it without changing something in JSOMR2MEI
ET.register_namespace('', 'http://www.music-encoding.org/ns/mei')
root = ET.fromstring(repair_xml(raw_xml))
tree = ET.ElementTree()
tree._setroot(root)

# this dict takes in any non-root element and returns its parent
parent_map = {c: p for p in tree.iter() for c in p}

# process image and transcript with ocropus and get aligned syllable bounding boxes
syls_boxes, image, lines_peak_locs = ocp.process(raw_image, transcript, wkdir_name='test')

# median vertical space between text lines, for later
med_line_spacing = np.median(np.diff(lines_peak_locs))

ns = {'id': '{http://www.w3.org/XML/1998/namespace}',
    'mei': '{http://www.music-encoding.org/ns/mei}'}
zones = root.findall('.//{}zone'.format(ns['mei']))
surface = root.findall('.//{}surface'.format(ns['mei']))[0]

# dictionary mapping id of every zone element to its bounding box information
id_to_bbox = {}
for zone in zones:
    id = zone.attrib[ns['id'] + 'id']
    id_to_bbox[id] = zone.attrib

syllable_elements = root.findall('.//{}syllable'.format(ns['mei']))
all_bboxes = []

cur_syllable = None         # current syllable element in tree being added to
prev_text = None            # last text found
prev_assigned_text = None   # last text assigned
elements_to_remove = []     # holds syllable elements containing duplicates
assign_lines = []           # lines b/w text and neumes, for visualization only

# iterate over syllable-level elements in the tree
for i, se in enumerate(syllable_elements):

    # get the neume associated with this syllable and the syllable's id
    neume = se[0]
    syl_id = se.attrib[ns['id'] + 'id']

    if not cur_syllable:
        cur_syllable = se

    # just in case something's gone horribly wrong
    assert 'neume' in neume.tag

    # get bounding boxes of every neume component in this syllable element, and find a bounding box
    # that contains all of the components
    neume_components = neume.findall(ns['mei'] + 'nc')
    bboxes = [id_to_bbox[nc.attrib['facs']] for nc in neume_components]

    lrx = max(int(bb['lrx']) for bb in bboxes)
    lry = max(int(bb['lry']) for bb in bboxes)
    ulx = min(int(bb['ulx']) for bb in bboxes)
    uly = min(int(bb['uly']) for bb in bboxes)

    # translate this bounding box downwards by half the height of a line
    # this should put well-positioned neumes right in the middle of the text they're associated with
    trans_lry = lry + med_line_spacing / 2
    trans_uly = uly + med_line_spacing / 2
    all_bboxes.append([ulx, uly, lrx, lry])

    # find text bounding boxes that intersect the translated neume bounding boxes
    colliding_syls = [s for s in syls_boxes
        if intersect(s[1], s[2], (ulx, trans_uly), (lrx, trans_lry)) > 0]

    # take just the leftmost text bounding box that was found
    if colliding_syls:
        leftmost_colliding_text = min(colliding_syls, key=lambda x: x[1][0])
        prev_assigned_text = leftmost_colliding_text
    else:
        leftmost_colliding_text = None

    # if there is no text OR if the found text is the same as last time then the neume being
    # considered here is linked to the previous syllable.
    if (not leftmost_colliding_text) or (leftmost_colliding_text == prev_text):
        cur_syllable.append(neume)
        elements_to_remove.append(se)

    # if the text found in the collision is new, then we're starting a new text syllable. register
    # it in the manifest section with a new zone and set the cur_syllable variable
    else:
        cur_syllable = se
        cur_syllable.text = leftmost_colliding_text[0]
        cur_syllable.append(neume)

        new_zone = ET.SubElement(surface, '{}zone'.format(ns['mei']))
        new_id = generate_id()
        cur_syllable.set('facs', new_id)

        new_zone.set(ns['id'] + 'id', new_id)
        new_zone.set('lrx', str(lrx))
        new_zone.set('lry', str(lry))
        new_zone.set('ulx', str(ulx))
        new_zone.set('uly', str(uly))

    # for visualization
    if prev_assigned_text:
        assign_lines.append([ulx, uly, prev_assigned_text[1][0], prev_assigned_text[1][1]])

    prev_text = leftmost_colliding_text


# remove syllable elements that just held neumes and are now duplicates
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
