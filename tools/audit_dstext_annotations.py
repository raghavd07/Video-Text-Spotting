from pathlib import Path
import xml.etree.ElementTree as ET

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "DSText" / "train"

def inspect_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()

    frames = root.findall("frame")
    objects = 0
    dont_care = 0
    invalid_polygons = 0
    missing_transcription = 0

    for frame in frames:
        for obj in frame.findall("object"):
            objects += 1

            text = obj.get("Transcription", "")
            if not text:
                missing_transcription += 1
            if "DONT#CARE" in text.upper():
                dont_care += 1

            points = obj.findall("Point")
            if len(points) < 4:
                invalid_polygons += 1
                continue

            for point in points:
                try:
                    float(point.attrib["x"])
                    float(point.attrib["y"])
                except (KeyError, ValueError):
                    invalid_polygons += 1
                    break

    return {
        "frames": len(frames),
        "objects": objects,
        "dont_care": dont_care,
        "missing_text": missing_transcription,
        "invalid_polygons": invalid_polygons,
    }

def main():
    files = sorted(DATASET.rglob("*_GT.xml"))
    print("DSText training annotation audit")
    print("XML files:", len(files))

    totals = {
        "frames": 0,
        "objects": 0,
        "dont_care": 0,
        "missing_text": 0,
        "invalid_polygons": 0,
    }
    errors = []

    for path in files:
        try:
            result = inspect_xml(path)
            for key in totals:
                totals[key] += result[key]
        except (ET.ParseError, OSError) as exc:
            errors.append((path, str(exc)))

    for key, value in totals.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("XML files with read errors:", len(errors))
    for path, error in errors[:10]:
        print(f"  {path}: {error}")

if __name__ == "__main__":
    main()
