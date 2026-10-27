import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from logo.data.convert_gt import build_gt
from logo.data.parsers import PARSERS, find_annotations, is_dont_care, parse_dstext, parse_icdar2013v
from logo.data.schema import load_gt, validate_gt

_spec = importlib.util.spec_from_file_location(
    "convert_annotations", Path(__file__).resolve().parents[1] / "tools" / "convert_annotations.py")
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)

XML_13V = """<?xml version="1.0" encoding="us-ascii"?>
<Frames>
  <frame ID="1" />
  <frame ID="2">
    <object Transcription="NO" ID="2001" Language="Catalan" Mirrored="Unmirrored" Quality="MODERATE">
      <Point x="762" y="169" /><Point x="762" y="187" /><Point x="781" y="169" /><Point x="781" y="187" />
    </object>
    <object Transcription="#Don't care#" ID="2002" Language="Catalan" Mirrored="Unmirrored" Quality="LOW">
      <Point x="0" y="0" /><Point x="0" y="5" /><Point x="5" y="0" /><Point x="5" y="5" />
    </object>
  </frame>
  <frame ID="3">
    <object Transcription="NO" ID="2001" Language="Catalan" Mirrored="Unmirrored" Quality="HIGH">
      <Point x="770" y="169" /><Point x="770" y="187" /><Point x="789" y="169" /><Point x="789" y="187" />
    </object>
    <object Transcription="#unreadable#" ID="3001" Quality="low">
      <Point x="1" y="1" /><Point x="1" y="9" /><Point x="9" y="1" /><Point x="9" y="9" />
    </object>
  </frame>
</Frames>
"""

XML_15V = """<?xml version="1.0" encoding="UTF-8"?>
<Frames>
  <frame ID="1">
    <object Transcription="##DONT#CARE##" ID="1001" Language="Spanish" Mirrored="Unmirrored" Quality="LOW" >
      <Point x="417" y="1" /><Point x="417" y="15" /><Point x="457" y="1" /><Point x="457" y="15" />
    </object>
    <object Transcription="###" ID="1002" Quality="MODERTE" >
      <Point x="10" y="10" /><Point x="10" y="20" /><Point x="30" y="10" /><Point x="30" y="20" />
    </object>
  </frame>
  <frame ID="2">
    <object Transcription="Café" ID="1003" Language="Spanish" Mirrored="Unmirrored" Quality="HIGH" >
      <Point x="50" y="50" /><Point x="50" y="60" /><Point x="90" y="50" /><Point x="90" y="60" />
    </object>
  </frame>
</Frames>
"""

XML_DSTEXT = """<?xml version="1.0" ?>
<Frames>
  <frame ID="1">
    <object ID="1" Transcription="TONY" category="scene" language="alphanumeric">
      <Point x="524" y="-2"/><Point x="549" y="-2"/><Point x="549" y="169"/><Point x="524" y="169"/>
    </object>
    <object ID="1" Transcription="TONY" category="scene" language="alphanumeric">
      <Point x="524" y="-2"/><Point x="549" y="-2"/><Point x="549" y="169"/><Point x="524" y="169"/>
    </object>
    <object ID="5" Transcription="##DONT#CARE##" category="scene" language="alphanumeric">
      <Point x="199" y="298"/><Point x="342" y="224"/><Point x="383" y="303"/><Point x="241" y="378"/>
    </object>
  </frame>
  <frame ID="2">
    <object ID="1" Transcription="TONY" category="scene" language="alphanumeric">
      <Point x="526" y="0"/><Point x="551" y="0"/><Point x="551" y="170"/><Point x="526" y="170"/>
    </object>
    <object ID="1" Transcription="PARK" category="scene" language="alphanumeric">
      <Point x="708" y="26"/><Point x="740" y="26"/><Point x="740" y="149"/><Point x="708" y="149"/>
    </object>
  </frame>
  <frame ID="4">
    <object ID="9" Transcription=" " category="scene" language="alphanumeric">
      <Point x="1" y="1"/><Point x="4" y="1"/><Point x="4" y="3"/><Point x="1" y="3"/>
    </object>
  </frame>
</Frames>
"""


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _all_objs(frames):
    return [o for objs in frames.values() for o in objs]


@pytest.mark.parametrize("ds,xml", [("icdar2013v", XML_13V), ("icdar2015v", XML_15V),
                                    ("dstext", XML_DSTEXT)])
def test_every_parser_quads_and_frames(tmp_path, ds, xml):
    x = _write(tmp_path / "v_GT.xml", xml)
    frames = PARSERS[ds](x)
    assert all(len(o["quad"]) == 8 and all(isinstance(v, float) for v in o["quad"])
               for o in _all_objs(frames))
    gt = validate_gt(build_gt("v", ds, frames, 0, 0, 0))
    assert list(gt["frames"]) == [str(i) for i in range(1, gt["num_frames"] + 1)]
    for objs in gt["frames"].values():
        ids = [o["track_id"] for o in objs]
        assert len(ids) == len(set(ids))
    assert PARSERS[ds](x) == frames  # deterministic -> IDs stable across runs


def test_icdar2013v_fields(tmp_path):
    frames = parse_icdar2013v(_write(tmp_path / "v_GT.xml", XML_13V))
    assert frames[1] == []
    o = frames[2][0]
    assert o["track_id"] == 2001 and o["text"] == "NO" and o["quality"] == "MODERATE"
    assert o["quad"] == [762.0, 169.0, 781.0, 169.0, 781.0, 187.0, 762.0, 187.0]
    assert frames[3][0]["track_id"] == 2001 and frames[3][0]["quality"] == "HIGH"
    assert frames[2][1]["dont_care"] and frames[3][1]["dont_care"] and frames[3][1]["quality"] == "LOW"


def test_icdar2015v_dont_care_quality_unicode(tmp_path):
    stats = {}
    frames = PARSERS["icdar2015v"](_write(tmp_path / "v_GT.xml", XML_15V), stats)
    assert [o["dont_care"] for o in frames[1]] == [True, True]
    assert frames[1][1]["quality"] == "MODERATE" and stats["quality_fixed"] == 1
    assert frames[2][0]["text"] == "Café" and not frames[2][0]["dont_care"]


def test_dstext_duplicates_and_null_quality(tmp_path):
    stats = {}
    frames = parse_dstext(_write(tmp_path / "v_GT.xml", XML_DSTEXT), stats)
    assert [o["track_id"] for o in frames[1]] == [1, 5]  # verbatim repeat dropped
    assert [(o["track_id"], o["text"]) for o in frames[2]] == [(1, "TONY"), (10, "PARK")]  # max 9 + 1
    assert stats == {"dup_dropped": 1, "dup_reid": 1}
    assert all(o["quality"] is None for o in _all_objs(frames))
    assert frames[1][0]["quad"][1] == -2.0 and frames[4][0]["dont_care"]


def test_dont_care_rule():
    for t in ("###", "##DONT#CARE##", "#Don't Care#", "#unreadable#", "", "  "):
        assert is_dont_care(t)
    assert not is_dont_care("????") and not is_dont_care("ALLERGIES?")


def test_find_annotations_case_insensitive(tmp_path):
    _write(tmp_path / "train" / "Sports" / "video_1_1_1_GT.xml", XML_13V)
    (tmp_path / "train" / "Sports" / "Video_1_1_1.mp4").write_bytes(b"")
    _write(tmp_path / "train" / "Video_2_1_1_GT.xml", XML_13V)
    out = find_annotations(tmp_path, "train")
    assert [(v, x.name, p.name if p else None) for v, x, p in out] == [
        ("Video_1_1_1", "video_1_1_1_GT.xml", "Video_1_1_1.mp4"), ("Video_2_1_1", "Video_2_1_1_GT.xml", None)]
    assert find_annotations(tmp_path, "test") == []


def test_tool_end_to_end(tmp_path, capsys):
    raw, frames_root, out = tmp_path / "raw", tmp_path / "frames", tmp_path / "processed"
    _write(raw / "dstext" / "train" / "Activity" / "Video_7_1_1_GT.xml", XML_DSTEXT)
    _write(raw / "dstext" / "train" / "Activity" / "Video_8_1_1_GT.xml", XML_DSTEXT)
    _write(raw / "dstext" / "test" / "Video_9_1_1_GT.xml", XML_DSTEXT)
    fdir = frames_root / "dstext" / "Video_7_1_1"
    fdir.mkdir(parents=True)
    for i in range(1, 6):
        cv2.imwrite(str(fdir / f"{i:06d}.jpg"), np.zeros((24, 40, 3), np.uint8))
    argv = ["--ds", "dstext", "--raw-root", str(raw), "--frames-root", str(frames_root),
            "--out", str(out), "--log", str(tmp_path / "log.txt")]
    assert ca.main(argv) == 0
    table = capsys.readouterr().out
    assert "size_flagged" in table and "dont_care" in table
    assert sorted(p.name for p in (out / "dstext").iterdir()) == ["Video_7_1_1.json", "Video_8_1_1.json"]
    a = load_gt(out / "dstext" / "Video_7_1_1.json")
    assert (a["width"], a["height"], a["num_frames"]) == (40, 24, 5)  # from frames, 5 > last xml id 4
    b = load_gt(out / "dstext" / "Video_8_1_1.json")
    assert (b["width"], b["height"], b["num_frames"]) == (0, 0, 4)
    row = ca.convert_dataset("dstext", raw, frames_root, out, True, ca.get_logger("t", None))
    assert row["videos"] == 3 and row["size_flagged"] == 2 and row["dont_care"] == 3 * 2
    assert row["instances"] == 3 * 5 and row["dup_dropped"] == 3 and row["dup_reid"] == 3
