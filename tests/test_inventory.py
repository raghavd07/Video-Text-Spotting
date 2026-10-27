import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inventory", Path(__file__).resolve().parents[1] / "tools" / "inventory.py")
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _make_raw(root: Path) -> None:
    for rel in ["icdar2013v/train/Video_1_1_1.mp4", "icdar2013v/train/Video_1_1_1_GT.xml",
                "icdar2013v/train/Video_2_1_1.mp4", "icdar2013v/train/video_2_1_1_GT.xml",
                "icdar2013v/test/Video_3_1_1.mp4", "icdar2013v/test/Video_9_9_9_GT.xml",
                "icdar2015v/train/Video_4_1_1.mp4", "icdar2015v/train/Video_4_1_1_GT.xml",
                "icdar2015v/train/Video_4_1_1_GT.txt", "icdar2015v/train/Video_4_1_1_GT_voc.txt",
                "icdar2015v/train/trainsetVocabulary.txt",
                "icdar2015v/test/Video_5_1_1.mp4", "icdar2015v/test/Video_5_1_1_GT_voc.txt",
                "dstext/train/Sports/Video_6_0_1.mp4", "dstext/train/Sports/Video_6_0_1_GT.xml",
                "dstext/test/Game/Video_7_0_2.mp4"]:
        _touch(root, rel)


def test_scan_counts_and_matching(tmp_path):
    _make_raw(tmp_path)
    r = inv.scan_root(tmp_path)

    t13 = r["icdar2013v"]["total"]
    assert (t13["num_videos"], t13["num_annotations"], t13["matched"]) == (3, 3, 2)
    assert t13["videos_without_ann"] == ["test/Video_3_1_1.mp4"]
    assert t13["anns_without_video"] == ["test/Video_9_9_9_GT.xml"]
    assert t13["case_mismatch"] == ["train/Video_2_1_1.mp4 <-> train/video_2_1_1_GT.xml"]

    s15 = r["icdar2015v"]["splits"]
    assert s15["train"]["matched"] == 1 and s15["train"]["num_aux"] == 2
    assert s15["test"]["num_annotations"] == 0
    assert s15["test"]["aux_only"] == ["test/Video_5_1_1.mp4"]

    ds = r["dstext"]["splits"]
    assert ds["train"]["matched"] == 1
    assert ds["test"]["videos_without_ann"] == ["test/Game/Video_7_0_2.mp4"]


def test_missing_dataset_folder(tmp_path):
    (tmp_path / "dstext").mkdir()
    r = inv.scan_root(tmp_path)
    assert r["icdar2013v"]["total"]["num_videos"] == 0
    assert "| icdar2013v | total | 0 | 28 |" in inv.render_table(r)


def test_append_report_creates_and_keeps_other_sections(tmp_path):
    rep = tmp_path / "docs" / "data_report.md"
    inv.append_report("### run1\n\nA\n", rep)
    text = rep.read_text(encoding="utf-8")
    assert text.startswith("# Data report\n") and "## Inventory\n\n### run1" in text

    rep.write_text(text + "\n## Other\n\nkeep\n", encoding="utf-8")
    inv.append_report("### run2\n\nB\n", rep)
    text = rep.read_text(encoding="utf-8")
    assert text.count("## Inventory") == 1
    assert text.index("### run1") < text.index("### run2") < text.index("## Other")
    assert text.rstrip().endswith("keep")


def test_main_no_write(tmp_path, capsys):
    _make_raw(tmp_path)
    rep = tmp_path / "r.md"
    assert inv.main(["--root", str(tmp_path), "--report", str(rep), "--no-write"]) == 0
    assert not rep.exists()
    assert "| dstext | test | 1 | 50 | 0 |" in capsys.readouterr().out
    assert inv.main(["--root", str(tmp_path / "nope")]) == 1
