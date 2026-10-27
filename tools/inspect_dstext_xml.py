from pathlib import Path
import xml.etree.ElementTree as ET

xml_file = next(Path("datasets/DSText/train").rglob("*_GT.xml"))
root = ET.parse(xml_file).getroot()
frame = root.find("frame")

print("XML file:", xml_file)
print("First frame attributes:", frame.attrib if frame is not None else None)

if frame is not None:
    for element in list(frame)[:10]:
        print("\nElement:", element.tag)
        print("Attributes:", element.attrib)
        print("Text:", repr(element.text))
        for child in list(element)[:8]:
            print("  Child:", child.tag)
            print("  Attributes:", child.attrib)
            print("  Text:", repr(child.text))
