import cv2
import numpy as np

from logo.data.convert_gt import build_gt, find_raw_videos, is_dont_care, parse_xml
from logo.data.schema import validate_gt

XML = """<?xml version="1.0" encoding="us-ascii"?>
<Frames>
  <frame ID="1">
    <object Transcription="NO" ID="1001" Language="Catalan" Mirrored="Unmirrored" Quality="MODERATE">
      <Point x="762" y="169" /><Point x="762" y="187" /><Point x="781" y="169" /><Point x="781" y="187" />
    </object>
    <object Transcription="#Don't care#" ID="1002" Quality="LOW">
      <Point x="0" y="0" /><Point x="5" y="0" /><Point x="5" y="5" /><Point x="0" y="5" />
    </object>
  </frame>
  <frame ID="2"/>
  <frame ID="3">
    <object ID="7" Transcription=" " category="scene" language="alphanumeric">
      <Point x="1" y="1"/><Point x="4" y="1"/><Point x="4" y="3"/><Point x="1" y="3"/>
    </object>
  </frame>
</Frames>
"""


def test_dont_care_variants():
    assert is_dont_care("##DONT#CARE##") and is_dont_care("#Don't care#")
    assert is_dont_care("   ") and is_dont_care("###") and is_dont_care("#unreadable#")
    assert not is_dont_care("ENCAS")


def test_parse_and_build(tmp_path):
    x = tmp_path / "v_GT.xml"
    x.write_text(XML, encoding="utf-8")
    frames = parse_xml(x)
    gt = validate_gt(build_gt("v", "icdar2013v", frames, 1280, 960, 4))
    assert gt["num_frames"] == 4 and list(gt["frames"]) == ["1", "2", "3", "4"]
    assert gt["frames"]["2"] == [] and gt["frames"]["4"] == []
    o = gt["frames"]["1"][0]
    assert o["track_id"] == 1001 and o["text"] == "NO" and o["quality"] == "MODERATE"
    assert o["quad"] == [762.0, 169.0, 781.0, 169.0, 781.0, 187.0, 762.0, 187.0]
    assert gt["frames"]["1"][1]["dont_care"] is True
    d = gt["frames"]["3"][0]
    assert d["dont_care"] is True and d["quality"] is None


def test_find_raw_videos_case_insensitive(tmp_path):
    folder = tmp_path / "train" / "Sports"
    folder.mkdir(parents=True)
    w = cv2.VideoWriter(str(folder / "Video_1_1_1.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 24))
    for _ in range(2):
        w.write(np.zeros((24, 32, 3), np.uint8))
    w.release()
    (folder / "video_1_1_1_GT.xml").write_text(XML, encoding="utf-8")
    out = find_raw_videos(tmp_path, "train")
    assert [(v, x.name) for v, _, x in out] == [("Video_1_1_1", "video_1_1_1_GT.xml")]
    assert find_raw_videos(tmp_path, "test") == []
