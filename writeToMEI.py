import xml.etree.cElementTree as ET

layer = ET.Element('layer')
syllable = ET.SubElement(layer, 'syllable')
ET.SubElement(syllable, "syl").text = "Splen"
syllable2 = ET.SubElement(layer, 'syllable')
ET.SubElement(syllable2, "syl").text = "dor"

tree = ET.ElementTree(layer)
ET.dump(tree)
