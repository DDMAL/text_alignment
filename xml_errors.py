# test file to mess around with the available MEI files to figure out why they aren't validating

import writeToMEI as wtm
import xml.etree.cElementTree as ET
import numpy as np


fname = 'salzinnes_{:02d}'.format(20)

with open('./mei/' + fname + '.mei', 'r') as f:
    raw_xml = f.read()

ns = {'id': '{http://www.w3.org/XML/1998/namespace}',
    'mei': '{http://www.music-encoding.org/ns/mei}'}

# forgive me for this but the xml output by pitchfinding has a namespace issue and this is the
# only way i can think of to correctly parse it without changing something in JSOMR2MEI
ET.register_namespace('', 'http://www.music-encoding.org/ns/mei')
try:
    root = ET.fromstring(raw_xml)
except ET.ParseError:
    root = ET.fromstring(wtm.repair_xml(raw_xml))

root = root[1]
tree = ET.ElementTree()
tree._setroot(root)


neume_components = root.findall('.//{}nc'.format(ns['mei']))
for nc in neume_components:
    if 'name' in nc.attrib.keys():
        nc.attrib['type'] = nc.attrib['name']
        del nc.attrib['name']

staves = root.findall('.//{}staff'.format(ns['mei']))
for st in staves:
    if st.attrib['line_positions']:
        del st.attrib['line_positions']
    if st.attrib['lines']:
        del st.attrib['lines']

graphic = root.findall('.//{}graphic'.format(ns['mei']))[0]
del graphic.attrib['{http://www.w3.org/1999/xlink}href']

# all_els = root.findall('.//')
# remove = []
# for el in all_els:
#     if '{http://www.w3.org/XML/1998/namespace}id' in el.attrib:
#         del el.attrib['{http://www.w3.org/XML/1998/namespace}id']
#     if 'custos' in el.tag or 'clef' in el.tag:
#         remove.append(el)
#
# # this dict takes in any non-root element and returns its parent
# parent_map = {c: p for p in tree.iter() for c in p}
#
# for el in remove:
#     parent_map[el].remove(el)

tree.write('testxml_clean_{}.xml'.format(fname))
