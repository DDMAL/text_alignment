import xml.etree.cElementTree as ET
import gamera.core as gc
gc.init_gamera()
import textSeqCompare as tsc
import ocropyTest as ocp


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

ns = {}

tree = ET.parse('salzinnes_mei_split/CF-011.mei')
root = tree.getroot()

zones = root.findall('.//{http://www.music-encoding.org/ns/mei}zone')
id_to_bbox = {}

for zone in zones:
    id = zone.attrib['{http://www.w3.org/XML/1998/namespace}id']
    id_to_bbox[id] = zone.attrib

syllable_elements = root.findall('.//{http://www.music-encoding.org/ns/mei}syllable')

for se in syllable_elements:
    # get the neume associated with this syllable
    neume = se[0]

    assert 'neume' in neume.tag
