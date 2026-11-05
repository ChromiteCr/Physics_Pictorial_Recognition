import json
import tarfile

from PIL import Image

from lab_detective.training.format import Example, write_jsonl
from lab_detective.training.package import build_package


def test_package_collects_referenced_images_and_manifest(tmp_path):
    root = tmp_path / "results"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "images" / "a.jpg")
    Image.new("RGB", (8, 8)).save(root / "images" / "unused.jpg")
    ex = Example("r1", "recognize", "photo", [
        {"role": "user", "content": [{"type": "image", "image": "images/a.jpg"}, {"type": "text", "text": "这是什么实验？"}]},
        {"role": "assistant", "content": "{}"}])
    write_jsonl(root / "train" / "recognize.jsonl", [ex])
    rep = build_package([root / "train" / "recognize.jsonl"], [], [], root, tmp_path / "out", tag="t")
    assert rep["n_images"] == 1 and rep["missing_images"] == 0
    with tarfile.open(rep["tarball"]) as tar:
        names = tar.getnames()
    assert "package_t/images/a.jpg" in names and "package_t/train.jsonl" in names
    assert not any(n.endswith("unused.jpg") for n in names)
    man = json.loads((tmp_path / "out" / "package_t" / "MANIFEST.json").read_text())
    assert man["counts"] == [{"split": "train", "task": "recognize", "source": "photo", "n": 1}]
