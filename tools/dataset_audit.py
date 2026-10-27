from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def audit_split(dataset, split):
    folder = DATASETS / dataset / split
    print(f"\n--- {dataset} / {split} ---")

    if not folder.exists():
        print("Folder not found")
        return

    videos = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    annotations = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and (
            p.name.lower().endswith("_gt.xml")
            or p.name.lower().endswith("_gt.txt")
            or p.name.lower().endswith("_gt_voc.txt")
        )
    )

    print(f"Videos: {len(videos)}")
    print(f"Annotation files: {len(annotations)}")

    annotation_stems = {
        p.name.lower().replace("_gt_voc.txt", "")
                    .replace("_gt.xml", "")
                    .replace("_gt.txt", "")
        for p in annotations
    }

    missing = [
        p for p in videos
        if p.stem.lower() not in annotation_stems
    ]

    print(f"Videos without matching annotation filename: {len(missing)}")
    for p in missing[:10]:
        print(f"  {p.relative_to(folder)}")

    for p in annotations[:3]:
        if p.suffix.lower() == ".xml":
            try:
                ET.parse(p)
                print(f"XML valid: {p.name}")
            except ET.ParseError as exc:
                print(f"XML parse error: {p.name}: {exc}")

if __name__ == "__main__":
    for dataset in ("DSText", "ICDAR2013", "ICDAR2015"):
        for split in ("train", "test"):
            audit_split(dataset, split)
